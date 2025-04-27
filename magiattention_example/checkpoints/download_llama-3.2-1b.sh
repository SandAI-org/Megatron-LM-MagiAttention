pip install modelscope
modelscope download --model LLM-Research/Llama-3.2-1B  # model stored in /root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/ by default
cp /root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/original/tokenizer.model ../
ln -s /root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/ ./Llama-3.2-1b # create soft link
cp ./Llama-3.2-1b/original/tokenizer.model ./Llama-3.2-1b