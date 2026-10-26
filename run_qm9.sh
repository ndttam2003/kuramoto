#!/bin/bash
    # --multi-gpu \
    # --num_processes=2  \
accelerate launch --cpu --num_processes=2 train_qm9.py \
    --data_path "/home/tam/code_base/my_egnn" \
    --project_name qm9_0 \
    --model_name EdgeGNN\
    --gnn_layer_type gat\
    --n_gnn_layers 3\
    --seed 593 \
    --epochs 500 \
    --warmup_iters 2 \
    --lr 1e-3 \
    --batchsize 128 \
    --num_workers 2 \
    --L 1 \
    --T 8 \
    --message_passing_type attn \
    --readout_type gat \
    --num_head 4 \
    --osc_dim 4 \
    --hidden_dim 128 \
    --task 0 \
    --gamma 1.0