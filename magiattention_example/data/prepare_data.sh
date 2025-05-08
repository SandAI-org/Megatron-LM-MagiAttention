TOKENIZER_MODEL=../checkpoints/Llama-3.2-1b

# prepare data for llama3.2
python3 ../tools/preprocess_data.py \
       --input openwebtext.json \
       --output-prefix llama_openwebtext \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model ${TOKENIZER_MODEL} \
       --workers 20 \
       --append-eod