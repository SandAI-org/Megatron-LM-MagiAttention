## Train llama-3-1b model 
We provide an example for you to train llama-3.2-1b with different megatron cp size. 

### prepare checkpoints
You can refer to the shell in dir checkpoints/
run download shell to download checkpoint from modelscope and run convert shell to transformer huggingface format checkpoint to megatron format.

### prepare data
We use openwebtext here to continue training megatron.
run prepare_data to download data from huggingface and preprocess data.

### Experiments
You can run train_llama_1b_from_scratch.sh to train model from scratch with different megatron cp size.
You can also run_llama_from_checkpoint.sh to continue train model with different megatron cp size.

### Results
Loss with global batch size 16 and 100000 training iters:
![alt text](./images/train_from_scratch.png)