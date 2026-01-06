import argparse
import json
from tqdm import tqdm
import sys
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from transformers import set_seed
from sample import evolve_agla_sampling

evolve_agla_sampling()
from augmentation import augmentation, multi_layer_augmentation
from lavis.models import load_model_and_preprocess
from torchvision import transforms


def setup_distributed():
    """初始化分布式环境"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # 设置当前GPU
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def eval_model_ddp(args):
    """使用DDP的推理函数"""
    # 初始化分布式
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"######### Start Evaluation with DDP, using {world_size} GPUs")

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # 加载模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # 移动到当前设备
    device = torch.device(f'cuda:{local_rank}')
    model = model.to(device)

    # 使用DDP包装模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 加载所有问题
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    # 创建分布式sampler
    sampler = DistributedSampler(
        questions,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )

    # 获取当前进程处理的数据
    indices = list(sampler)
    current_questions = [questions[i] for i in indices]

    # 每个进程写入自己的结果文件
    base_name = os.path.splitext(args.answers_file)[0]
    ext = os.path.splitext(args.answers_file)[1]
    per_rank_file = f"{base_name}_rank{rank}{ext}"

    os.makedirs(os.path.dirname(per_rank_file), exist_ok=True)
    ans_file = open(per_rank_file, "w")

    # 加载BLIP模型
    model_itm, vis_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=device, is_eval=True
    )
    loader = transforms.Compose([transforms.ToTensor()])

    # 进度条：只在rank 0显示
    if rank == 0:
        iterator = tqdm(current_questions, desc="Processing")
    else:
        iterator = current_questions

    for line in iterator:
        idx = line["question_id"]
        image_file = line["image"]
        question = line["text"]
        cur_prompt = question

        if model.module.config.mm_use_im_start_end:  # 注意：DDP包装后要用 .module
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            device)

        raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        raw_image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

        if args.use_agla:
            tensor_image = loader(raw_image.resize((384, 384)))
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            question_text = text_processors["eval"](question)
            tokenized_text = model_itm.tokenizer(question_text, padding='longest', truncation=True,
                                                 return_tensors="pt").to(device)

            augmented_image = augmentation(
                image, question_text, tensor_image, model_itm, tokenized_text, raw_image,
                model.module.model.vision_tower.vision_tower,  # 注意：DDP包装后要用 .module
                model.module.model.vision_tower.image_processor
            )

            # 保存增强后的图像
            output_folder = "./augmented_images"
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"rank{rank}_{image_file}")
            augmented_image.save(output_path)

            image_tensor = image_processor.preprocess(augmented_image, return_tensors='pt')['pixel_values'][0]
        else:
            image_tensor = None

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.module.generate(  # 注意：DDP包装后要用 .module
                input_ids,
                images=raw_image_tensor.unsqueeze(0).half().to(device),
                images_cd=(image_tensor.unsqueeze(0).half().to(device) if image_tensor is not None else None),
                cd_alpha=args.alpha,
                cd_beta=args.beta,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # 写入结果
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "model_id": model_name,
            "image": image_file,
            "metadata": {"rank": rank}
        }) + "\n")
        ans_file.flush()

    ans_file.close()

    # 等待所有进程完成
    dist.barrier()

    # 主进程合并所有结果
    if rank == 0:
        merge_results(args, world_size)
        print(f"合并完成！结果保存在 {args.answers_file}")

    cleanup_distributed()


def merge_results(args, world_size):
    """合并所有进程的结果"""
    base_name = os.path.splitext(args.answers_file)[0]
    ext = os.path.splitext(args.answers_file)[1]

    all_results = []
    for rank in range(world_size):
        per_rank_file = f"{base_name}_rank{rank}{ext}"
        if os.path.exists(per_rank_file):
            with open(per_rank_file, 'r') as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))
            # 删除临时文件
            os.remove(per_rank_file)

    # 按question_id排序
    all_results.sort(key=lambda x: x['question_id'])

    # 写入最终结果
    with open(args.answers_file, 'w') as f:
        for result in all_results:
            json.dump(result, f)
            f.write('\n')

    print(f"总共有 {len(all_results)} 条结果")


def eval_model(args):
    """单GPU推理函数（原有逻辑）"""
    print("######### Start Evaluation (Single GPU).")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    device = 'cuda:0'

    model_itm, vis_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=device, is_eval=True
    )
    loader = transforms.Compose([transforms.ToTensor()])

    for line in tqdm(questions):
        # ... 原有单GPU推理逻辑 ...
        idx = line["question_id"]
        image_file = line["image"]
        question = line["text"]
        cur_prompt = question

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        raw_image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

        if args.use_agla:
            tensor_image = loader(raw_image.resize((384, 384)))
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            question_text = text_processors["eval"](question)
            tokenized_text = model_itm.tokenizer(question_text, padding='longest', truncation=True,
                                                 return_tensors="pt").to('cuda')

            augmented_image = augmentation(
                image, question_text, tensor_image, model_itm, tokenized_text, raw_image,
                model.model.vision_tower.vision_tower,
                model.model.vision_tower.image_processor
            )

            output_folder = "./augmented_images"
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, image_file)
            augmented_image.save(output_path)

            image_tensor = image_processor.preprocess(augmented_image, return_tensors='pt')['pixel_values'][0]
        else:
            image_tensor = None

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=raw_image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor.unsqueeze(0).half().cuda() if image_tensor is not None else None),
                cd_alpha=args.alpha,
                cd_beta=args.beta,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "model_id": model_name,
            "image": image_file,
            "metadata": {}
        }) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/yzh/cs285/AGLA/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/yzh/cs285/AGLA/pope_source/val2014")
    parser.add_argument("--question-file", type=str,
                        default="/home/yzh/cs285/AGLA/data/POPE/coco/coco_pope_adversarial.json")
    parser.add_argument("--answers-file", type=str, default="/home/yzh/cs285/AGLA/eval/output/test.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--use_agla", action='store_true', default=True)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ddp", action='store_true', help="使用DDP进行多GPU推理")
    parser.add_argument("--local_rank", type=int, default=0, help="DDP使用的本地rank")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.use_ddp and torch.cuda.device_count() > 1:
        eval_model_ddp(args)
    else:
        eval_model(args)