#!/bin/bash

# 定义GPU ID列表
gpu_ids=(0 1 2 3 4 5 6 7)

# 起始端口号
start_port=7100

# 遍历每个GPU ID并启动相应的进程
for i in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[$i]}"
    port=$((start_port + i))

    echo "Starting server on GPU $gpu_id and port $port"
    python cosyvoice_fastapi_serve_ws.py --port $port --gpu_id $gpu_id &
    pid=$!
    echo "PID of the process: $pid"
    pids+=($pid)  # 保存所有子进程的PID
done

# 等待所有子进程结束
wait ${pids[@]}