#!/bin/bash

# 定义日志目录
LOG_DIR="../logger/magi_v0.11/llama-3.2-1b/fix_rope"

# 检查日志目录是否存在，不存在则创建
mkdir -p "$LOG_DIR"

# 定义要运行的参数列表
params=(1 2 4 8)

# 循环执行并重定向日志
for num in "${params[@]}"; do
    log_file="${LOG_DIR}/cp_${num}.log"
    echo "Running with $num GPUs, logging to $log_file"
    bash run_llama_from_checkpoint.sh "$num" >& "$log_file"
done

echo "All tasks completed."