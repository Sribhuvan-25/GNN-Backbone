#!/bin/bash
# Script to run the Methane GNN pipeline

echo "===== Starting Methane GNN Pipeline ====="

# Activate your conda environment if needed
# source activate your_env_name

# Set CUDA visible devices if needed
# export CUDA_VISIBLE_DEVICES=0

# Run the methane GNN pipeline with specified parameters
python MethaneGNN.py \
    --data_path ../Data/New_data.csv \
    --k_neighbors 5 \
    --mantel_threshold 0.05 \
    --model_type gat \
    --model_architecture default \
    --hidden_dim 128 \
    --num_layers 4 \
    --dropout_rate 0.3 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --weight_decay 1e-4 \
    --num_epochs 300 \
    --patience 30 \
    --num_folds 5 \
    --save_dir ./methane_results \
    --visualize_graphs True

echo "===== Pipeline Completed =====" 