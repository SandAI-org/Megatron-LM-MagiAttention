#!/bin/bash

# Runs the "175B" parameter model
rm -rf checkpoint/

export CUDA_DEVICE_MAX_CONNECTIONS=1

context_parallel_size=$1
GPUS_PER_NODE=$context_parallel_size
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

#CHECKPOINT_PATH=$1 #<Specify path>
#TENSORBOARD_LOGS_PATH=$2 #<Specify path>
#VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
#MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
#DATA_PATH=$5 #<Specify path and file prefix>_text_document

# change your checkpoint_path and logger path
CHECKPOINT_LOAD_PATH=../checkpoints/llama-3.2-1b
CHECKPOINT_SAVE_PATH=../checkpoints/Llama-3.2-1b-new
TENSORBOARD_LOGS_PATH=../logger/megatron_v0.11/Llama-3.2-1b/te_cp_$context_parallel_size
TOKENIZER_MODEL=../checkpoints/Llama-3.2-1b

#TENSORBOARD_LOGS_PATH=../logger/megatron_v0.11/const_lr/cp_$context_parallel_size
rm -rf $TENSORBOARD_LOGS_PATH

#VOCAB_FILE=../prepare_dataset/gpt2-vocab.json
#MERGE_FILE=../prepare_dataset/gpt2-merges.txt
DATA_PATH=../prepare_dataset/llama_openwebtext_text_document


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
    --global-batch-size 1
    --train-iters 1000
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine
    #--lr-decay-style constant
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
    --lr-decay-iters 430000 
    #--transformer-impl local
    --exit-on-missing-checkpoint
    --use-checkpoint-args
    --no-load-optim
    --no-load-rng
    --untie-embeddings-and-output-weights
    --normalization RMSNorm
    --position-embedding-type rope
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --disable-bias-linear
    --transformer-impl transformer_engine
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
    #--vocab-file $VOCAB_FILE 
    #--merge-file $MERGE_FILE 
    --split 94,5,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_SAVE_PATH 
    --load $CHECKPOINT_LOAD_PATH
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}