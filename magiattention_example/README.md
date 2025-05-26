## Train llama-3-1b model 
We provide an example for you to train llama-1b model from scratch with different te cp_size. 

### prepare checkpoints
You can refer to the shell in dir ./checkpoints to download checkpoint in huggingface format and convert checkpoint to megatron format.

This is not necessary for training from scratch.

### prepare data
We use openwebtext(https://huggingface.co/datasets/Skylion007/openwebtext).

You can run refer to shell in dir ./data to download data from huggingface and preprocess data.

### Experiments
You can run train_llama_1b_from_scratch.sh to train model from scratch with different megatron cp size.
You can also run run_llama_from_checkpoint.sh to continue train model from checkpoint with different megatron cp size.

### Results
Loss with global batch size 16 and 100000 training iters:
![alt text](./images/train_from_scratch.png)

