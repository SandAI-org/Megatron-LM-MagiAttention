## train llama-3.2-1b with checkpoint and magiattention
We provide an example for you to train llama-3-1b with magiattention with different cp size. 

### prepare checkpoints
You can refer to the shell in dir ../checkpoints/ and
run prepare_llama-3.2-1b_checkpoint.sh to download checkpoint from modelscope and convert checkpoint from huggingface format to megatron format.

This is not necessary for training from scratch.

### prepare data
We use openwebtext(https://huggingface.co/datasets/Skylion007/openwebtext).

You can run prepare_data.sh int ./data to download data from huggingface and preprocess data.

### intergrate with magiattention
We intergrate magiattention with te and local transformer inplementation.
Main changes:
- add pretrain_llama.py which changes from pretrain_gpt.py.
- replace core_attention with magi_attention in gpt_layer_specs.py for local transformer_impl.
- replace forward of TEDoctProductAttention with magi_attention for Te transformer_engine transformer_impl.
- pass magi_attention_key through gpt model forward pass.
- replace get_pos_emb_on_this_cp_rank with get_pos_emb_on_this_cp_rank_magi for rope.

### experiment results
You can run train_llama_1b_from_scratch.sh to train llama-1b from scratch with magiattention.

Loss with global batch size 16 and 100000 training iters:
 ![alt text](./images/train_from_scratch.png)

Feel free to open any issue in this repo if you have any question!