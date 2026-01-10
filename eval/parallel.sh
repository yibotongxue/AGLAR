#! /bin/bash

dataset_name='coco' # 'aokvqa' 'gqa'
image_folder='/root/autodl-tmp/val2014' # '/workspace/data/gqa'

run () {
    seed=$1
    type=$2
    use_entropy=$3
    if [ "$use_entropy" = true ] ; then
        agla_args="--use_agla --use-entropy --alpha 2 --beta 0.5"
        answer="answers_entropy"
    else
        agla_args="--use_agla --alpha 2 --beta 0.5"
        answer="answers"
    fi
    python run_llava.py \
--question-file ../data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ../output/llava_${dataset_name}_pope_${type}_${answer}_seed${seed}.jsonl \
--seed $seed \
$agla_args
}

arg_list=()
for seed in 1
do
for type in 'random' 'popular' 'adversarial'
do
arg_list+=("$seed" "$type" "true")
# arg_list+=("$seed" "$type" "false")
done
done

gpu_index=$1
all_gpu_ids=(0) # 修改为可用的GPU ID列表
gpu_cnt=${#all_gpu_ids[@]}
echo "Running on GPU ${gpu_index}"
export CUDA_VISIBLE_DEVICES=${all_gpu_ids[$gpu_index]}
# 仅在GPU${gpu_index}上运行arg_list[i]，i%3=0且i/3%all_gpus=gpu_index
for i in "${!arg_list[@]}"
do
    if [[ $(( (i/3) % gpu_cnt )) -eq $gpu_index ]] && [[ $(( i % 3 )) -eq 0 ]]; then
        # echo "${i}"
        run "${arg_list[i]}" "${arg_list[i+1]}" "${arg_list[i+2]}"
    fi
done
