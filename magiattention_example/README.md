# Training LLaMA-3B with MagiAttention

This repository provides an example of training the LLaMA-3B model using MagiAttention. Below are the steps to get started.

## Prerequisites

### Prepare Checkpoints

You can use the shell script in the `magiattention/checkpoints/` directory to download and convert the checkpoint from ModelScope to the Megatron format.

> **Note**: The `tokenizer.model` file is required for training from scratch.

```bash
cd magiattention/checkpoints/
./prepare_llama-3.2-1b_checkpoint.sh
```

### Prepare Data

We use the [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset for training, following the Megatron-LM approach.

```bash
cd data/
./prepare_data.sh
```

## Integration with MagiAttention

### Modifications

We have integrated MagiAttention with `transformer_engine` and the local transformer implementation. The main changes are as follows:

1. **Modify `pretrain_gpt.py`**:
   - Update the `get_batch` function to use `dispatch_along_cp_rank`.
   - Add the `prepare_data` and `prepare_magi_attention` functions.

   ```diff
   def get_batch(data_iterator):
       """Generate a batch."""
       batch = get_batch_on_this_tp_rank(data_iterator)
   -   batch = get_batch_on_this_cp_rank(batch)
   +   batch = dispatch_along_cp_rank(batch)
       return batch.values()

   + def dispatch_along_cp_rank(batch: Dict[str, Any]):
       ...
       # squash batch dim for token and label and compute pad_size
   +   tokens, labels, cu_seqlens_q, cu_seqlens_k, pad_size = prepare_data(tokens, labels)
   +   input, dist_attn_runtime_key = prepare_magi_attention(
   +       tokens, cu_seqlens_q, cu_seqlens_k, pad_size, mpu.get_context_parallel_group())
       # update dict key
       ...
       return batch

   + def prepare_magi_attention(input, cu_seqlens_q, cu_seqlens_k, pad_size, cp_group):
   +   dist_attn_config = DistAttnConfig()
   +   ...
   +   x_padded, dist_attn_runtime_key = magi_attn_varlen_dispatch(
   +       input,
   +       cu_seqlens_q,
   +       cu_seqlens_k,
   +       head_dim=head_dim,
   +       pad_size=pad_size,
   +       cp_group=cp_group,
   +       causal=True,
   +       dist_attn_config=dist_attn_config,
   +   )
   +   return x_padded, dist_attn_runtime_key

   def forward_step(data_iterator, model: GPTModel):
       ...
       with stimer(bdata=True):
   -       tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
   -           data_iterator)
   +       tokens, labels, loss_mask, attention_mask, position_ids, key = get_batch(
   +           data_iterator)

       ...
   -   output_tensor = model(tokens, position_ids, attention_mask,
   -                         labels=labels)
   +   output_tensor = model(tokens, position_ids, attention_mask,
   +                        labels=labels, magi_attention_key=key)
       return output_tensor, partial(loss_func, loss_mask)
   ```

2. **Add `magi_attention.py`**:
   - Add the MagiAttention implementation to `megatron/core/transformer/magi_attention.py`.

3. **Replace `core_attention` with `magi_attention`**:
   - Replace `core_attention` with `magi_attention` in `megatron/core/models/gpt/gpt_layer_specs.py`.

4. **Update `TEDotProductAttention`**:
   - Replace the forward function of `TEDotProductAttention` with the MagiAttention forward function.

   ```diff
   class TEDotProductAttention(te.pytorch.DotProductAttention):
       def __init__():
           ...
   +       self.magi_attention = MagiAttention(config=config, 
   +                                           layer_number=layer_number,
   +                                           attn_mask_type=attn_mask_type,
   +                                           attention_type=attention_type,
   +                                           attention_dropout=attention_dropout,
   +                                           softmax_scale=softmax_scale)
           ...

       def forward():
   +       core_attn_out = self.magi_attention.forward(query=query, key=key, value=value, attention_mask=attention_mask, attention_bias=attention_bias, packed_seq_params=None, magi_attention_key=magi_attention_key)
   +       return core_attn_out
           '''
   -           core_attn_out = super().forward()
           '''
   ```

5. **Pass `magi_attention_key` through the GPT model forward pass**.

6. **Replace `get_pos_emb_on_this_cp_rank` with `get_pos_emb_on_this_cp_rank_magi`**:

   ```diff
   - def get_pos_emb_on_this_cp_rank(pos_emb: Tensor, seq_dim: int) -> Tensor:
   -    ...

   + def get_pos_emb_on_this_cp_rank_magi(pos_emb: Tensor, magi_attention_key) -> Tensor:
   +     from magi_attention.api import get_position_ids
   +
   +     cp_idx = get_position_ids(magi_attention_key)
   +     pos_emb = pos_emb[cp_idx]
   +
   +     return pos_emb
   ```

### Shells

- `train_llama_1b_from_scratch.sh`: Train the LLaMA-1B model from scratch.
- `resume_from_checkpoint.sh`: Resume training the LLaMA-1B model from a Megatron checkpoint.
- `run_llama_from_checkpoint.sh`: Train the LLaMA-1B model from a checkpoint converted from the Hugging Face format.

## Experiments

You can run `train_llama_1b_from_scratch.sh` to train the LLaMA-1B model from scratch with MagiAttention.

### Training Settings

- **Model Size**: LLaMA-1B
  - **Number of Layers**: 16
  - **Hidden Size**: 2048
  - **Number of Attention Heads**: 32
  - **Group Query Attention**: Yes
  - **Number of Query Groups**: 8
- **Sequence Length**: 8192
- **Context Parallel Size**: cp1/2/4/8 (MagiAttention vs TE Ring Attention) with global batch size 16.
- **Training Iterations**: 100000

### Results

MagiAttention aligns well with TE Ring Attention.

![Training from Scratch](./images/train_from_scratch.png)

Feel free to open an issue in this repository if you have any questions!