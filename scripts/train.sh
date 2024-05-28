set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATA=$1
MODEL_NAME=$2
OUTPUT_DIR=output/${MODEL_NAME}

mkdir -p ${OUTPUT_DIR}
cp "$0" ${OUTPUT_DIR}/
(deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port $((30000+$RANDOM%10000)) FastChat/fastchat/train/train_lora.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5-16k  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path ${DATA} \
    --run_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --fp16 True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --q_lora False \
    --deepspeed deepspeed_lora.json \
    --gradient_checkpointing True \
    --flash_attn False | tee ${OUTPUT_DIR}/log) 3>&1 1>&2 2>&3 | tee ${OUTPUT_DIR}/err
