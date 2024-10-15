
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=4
NUM_TRAIN_EPOCHS=1
MAX_SOURCE_LENGTH=128
MAX_TARGET_LENGTH=512
SPACE_TAG=aya_en_zh_300_rk+1_len64
LANG_LIST=en,zh
MODEL_ID=llama3

PUSH_END_LAYER=30
GRAD_ACCUMULATION_STEPS=8
LEARNING_RATE=1e-5
PULL_WEIGHT="1.0,1.0"
PUSH_WEIGHT=1.0
RETAIN_WEIGHT=1.0

# Environment setup
BACKBONE=/path/to/model
OUTPUT_DIR_BASE=result/$MODEL_ID
DATA_DIR=Manipulation/datasets/language_pull_data_alpaca_safe_200
DS_CONFIG=Manipulation/ds_config/stage2.config

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
port=$(shuf -i25000-30000 -n1)




ACTUAL_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GRAD_ACCUMULATION_STEPS))
RUN_NAME=edit_lr${LEARNING_RATE}_cosine_bs$((ACTUAL_BATCH_SIZE))_push$((PUSH_WEIGHT_VALUE))_pull$(echo $PULL_WEIGHT | cut -d',' -f2)_retain$((RETAIN_WEIGHT))_push_end_layer_$((PUSH_END_LAYER + 1))_epo$((NUM_TRAIN_EPOCHS))_data200_$SPACE_TAG
OUTPUT_DIR="$OUTPUT_DIR_BASE/$RUN_NAME"

echo "Starting training with the following parameters:"
echo "PUSH_END_LAYER: $PUSH_END_LAYER"
echo "GRAD_ACCUMULATION_STEPS: $GRAD_ACCUMULATION_STEPS"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "PULL_WEIGHT: $PULL_WEIGHT"

deepspeed --num_gpus=1 --master_port $port Manipulation/src/run.py \
    --model_name_or_path $BACKBONE \
    --output_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --bf16 \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length $MAX_TARGET_LENGTH \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --overwrite_output_dir \
    --save_strategy no \
    --save_steps 50 \
    --evaluation_strategy no \
    --eval_steps 200 \
    --prediction_loss_only True \
    --logging_strategy steps \
    --logging_steps 1 \
    --deepspeed $DS_CONFIG \
    --push_end_layer $PUSH_END_LAYER \
    --retain_weight $RETAIN_WEIGHT \
    --space_tag $SPACE_TAG \
    --lang_list $LANG_LIST \
    --pull_weight $PULL_WEIGHT \
    --push_weight $PUSH_WEIGHT \
    --model_id $MODEL_ID \
    --load_best_model_at_end True