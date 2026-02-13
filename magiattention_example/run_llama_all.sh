#!/bin/bash

LOG_DIR="../logger/magi_v0.11/llama-3.2-1b/fix_rope"

mkdir -p "$LOG_DIR"

params=(1 2 4 8)

for num in "${params[@]}"; do
    log_file="${LOG_DIR}/cp_${num}.log"
    echo "Running with $num GPUs, logging to $log_file"
    bash run_llama_from_checkpoint.sh "$num" >& "$log_file"
done

echo "All tasks completed."