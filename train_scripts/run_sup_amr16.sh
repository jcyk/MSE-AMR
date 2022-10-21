#!/bin/bash
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
set -e

parser=best_fr
languages=ar
for seed in 1 2 3 4 5; do
for i in 3; do
data=${parser}${languages}balancedxrandom${i}
for model in bert-uncased-large; do
OUTPUT_DIR=result/${data}amr_16my-sup-simcse-${model}${seed}
# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=4

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python3 -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --use_amr \
    --model_name_or_path ${model} \
    --train_file data/nli.${data}.csv \
    --path_to_amr backup/${parser}/sts-test-pred.txt \
    --seed ${seed} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --proj_mlp_type simcse_sup \
    --after_projector \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16

python3 simcse_to_huggingface.py --path ${OUTPUT_DIR}
python3 evaluation.py --model_name_or_path ${OUTPUT_DIR} --task_set sts --pooler simcse_sup  > ${OUTPUT_DIR}/sts.old.log
python3 evaluation.py --model_name_or_path ${OUTPUT_DIR} --task_set sts --pooler simcse_sup --use_amr --path_to_amr backup/${parser}/sts-test-pred.txt > ${OUTPUT_DIR}/sts.log
done
done
done

bash run_sup_amr16fr.sh 
