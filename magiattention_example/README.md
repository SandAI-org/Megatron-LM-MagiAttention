## Train llama-3-1b model 
We provide an example for you to train llama-1b model from scratch with different te cp_size. 

### prepare checkpoints
You can refer to the shell in dir ./checkpoints to download checkpoint in huggingface format and convert checkpoint to megatron format.

This is not necessary for training from scratch(tokenizer.model is needed).

### prepare data
We use openwebtext(https://huggingface.co/datasets/Skylion007/openwebtext) as our dataset.

You can run refer to shell in dir ./data to download data from huggingface and preprocess data.

### Experiments
You can run train_llama_1b_from_scratch.sh to train model from scratch with different megatron cp size.
You can also run run_llama_from_checkpoint.sh to continue train model from checkpoint with different megatron cp size.

### Experiments
 You can run ./magiattention_example/train_llama_1b_from_scratch.sh to train llama-1b from scratch with magiattention.
 training_settings:
 - model-size: llama-1b
     - num-layers: 16
     - hidden-size: 2048
     - num-attention-heads: 32
     - group-query-attention
     - num-query-groups: 8
 - seqlen: 8192
 - context_parallel_size: cp1/2/4/8(magiattention vs te ring attention) with global batch size 16.
 - train_iters: 100000

 Results:
 MagiAttention aligns well with te ring attention.
![alt text](./images/train_from_scratch.png)

