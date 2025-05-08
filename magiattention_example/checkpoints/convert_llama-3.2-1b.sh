TP=1
HF_FORMAT_DIR=/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B
MEGATRON_FORMAT_DIR=./llama-3.2-1b
TOKENIZER_MODEL=/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/

python ../tools/checkpoint/convert.py \
    --bf16 \
    --model-type GPT \
    --loader llama_mistral \
    --saver core \
    --checkpoint-type hf \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --model-size llama3 \
