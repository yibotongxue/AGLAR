dataset_name='coco' # 'aokvqa' 'gqa'
image_folder='/home/yzh/cs285/AGLA/pope_source/val2014' # '/workspace/data/gqa'
export MASTER_ADDR=localhost
export MASTER_PORT=12345
for seed in 1 2 3
do
for type in 'random' 'popular' 'adversarial'
do
torchrun \
  --nproc_per_node=8 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  run_llava_ddp.py \
  --use-ddp \
  --question-file ../data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
  --image-folder ${image_folder} \
  --answers-file ../output/llava_${dataset_name}_pope_${type}_answers_agla_seed${seed}.jsonl \
  --use_agla \
  --alpha 2 \
  --beta 0.5 \
  --seed $seed
done
done