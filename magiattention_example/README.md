## train llama-3.2-1b with checkpoint and magiattention
We provide an example for you to train llama-3.2-1b with different cp size. 

### prepare checkpoints
You can refer to the shell in dir ../checkpoints/ and
run prepare_llama-3.2-1b_checkpoint.sh to download checkpoint from modelscope and run convert checkpoint from huggingface format to megatron format.

### prepare data
We use openwebtext(https://huggingface.co/datasets/Skylion007/openwebtext) to continue training llama-3.2-1b.

You can run prepare_data.sh int ./data to download data from huggingface and preprocess data.

### intergrate with magiattention
We intergrate magiattention with te and local transformer inplementation.
You can compare the current commit with commit 97cdef4b68464e20c722fa4a59d62240dea7d93b to find out the changes.
Main changes:
- add pretrain_llama.py which come from pretrain_gpt.py
- replace core_attention with magi_attention.
- pass magi_attention_key through model forward pass.
- replace get_pos_emb_on_this_cp_rank for rope.

### experiments
You can run run_llama_all.sh to do experiments with magiattention cp1/2/4/8 and observe how the loss change.

results(1000 train steps with bs1 and te transformer inplementation):
![alt text](image.png)