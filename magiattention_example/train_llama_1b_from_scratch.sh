#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=9010
[ -z "$NUM_NODES" ] && NUM_NODES=1
context_parallel_size=$1
GPUS_PER_NODE=8

# change your checkpoint_path and logger path
CHECKPOINT_SAVE_PATH=/your_checkpoint_path
TENSORBOARD_LOGS_PATH=/your_tensorboard_path
TOKENIZER_MODEL=./Llama-3.2-1b   # your tokenizer_model
rm -rf $TENSORBOARD_LOGS_PATH

DATA_PATH=./data/llama_openwebtext_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 16
    --hidden-size 2048
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --seq-length 8192
    --max-position-embeddings 131072 
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --rotary-base 500000
    --rotary-percent 1.0
    --use-rope-scaling
    --swiglu
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 16
    --train-iters 100000
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
    --lr-decay-iters 430000 
    --transformer-impl transformer_engine # use te by default
    --untie-embeddings-and-output-weights
    --normalization RMSNorm
    --position-embedding-type rope
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --disable-bias-linear
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size $context_parallel_size
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --split 94,5,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 50000
    --eval-interval 10000
    --save $CHECKPOINT_SAVE_PATH 
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

