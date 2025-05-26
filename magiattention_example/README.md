## train llama-3-1b with magiattention
We provide an example for you to train llama-3-1b with magiattention with different cp size. 

### Prepare checkpoints
You can refer to the shell in dir ../checkpoints/ and
run prepare_llama-3.2-1b_checkpoint.sh to download checkpoint from modelscope and convert checkpoint from huggingface format to megatron format.

tokenizer.model is necessary for training from scratch.

### Prepare data
We use openwebtext(https://huggingface.co/datasets/Skylion007/openwebtext) dataset.

You can run prepare_data.sh in ./data to download data from huggingface and preprocess data.

### Intergrate with magiattention
We intergrate magiattention with transformer_engine and local transformer inplementation.
Main changes:
- add pretrain_gpt.py which changes from pretrain_gpt.py.
    - mainly change dispatch_along_cp_rank, prepare_data and prepare_magi_attention function.
- replace core_attention with magi_attention in gpt_layer_specs.py for local transformer_impl.
- replace forward function of TEDoctProductAttention with magi_attention forward function for Te transformer_engine transformer_impl.
- pass magi_attention_key through gpt model forward pass.
- replace get_pos_emb_on_this_cp_rank with get_pos_emb_on_this_cp_rank_magi for rope.

### Experiments
You can run train_llama_1b_from_scratch.sh to train llama-1b from scratch with magiattention.
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

Feel free to open any issue in this repo if you have any question!