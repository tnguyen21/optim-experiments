#!/bin/bash

# Multi-seed experiment script with optimized configurations
# Uses optimizer-specific configs with optimal learning rates from LR sweep
# Usage: ./scripts/run_multi_seed_experiment.sh

set -e  # Exit on any error

# Configuration
SEEDS=(42 123 456 314 2718)

# Optimizer-specific configurations with optimal LRs
declare -A OPTIMIZER_CONFIGS
OPTIMIZER_CONFIGS["sgd"]="config/sgd_optimal_config.yaml"
OPTIMIZER_CONFIGS["adamw"]="config/adamw_optimal_config.yaml"
OPTIMIZER_CONFIGS["muon"]="config/muon_optimal_config.yaml"

OPTIMIZERS=("sgd" "adamw" "muon")

echo "Starting multi-seed optimizer comparison experiment..."
echo "Using optimizer-specific configs with optimal learning rates:"
for opt in "${OPTIMIZERS[@]}"; do
    echo "  $opt: ${OPTIMIZER_CONFIGS[$opt]}"
done
echo "Seeds: ${SEEDS[@]}"
echo

# Function to run a single experiment
run_experiment() {
    local optimizer=$1
    local seed=$2
    local config_file=${OPTIMIZER_CONFIGS[$optimizer]}
    
    echo "Running experiment: optimizer=$optimizer, seed=$seed"
    echo "Config: $config_file"
    
    python scripts/run_experiment.py \
        --config $config_file \
        --seed $seed
    
    echo "Completed: optimizer=$optimizer, seed=$seed"
    echo "----------------------------------------"
}

# Run all combinations
total_experiments=$((${#SEEDS[@]} * ${#OPTIMIZERS[@]}))
current_experiment=0

for optimizer in "${OPTIMIZERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        current_experiment=$((current_experiment + 1))
        echo "Experiment $current_experiment/$total_experiments"
        run_experiment $optimizer $seed
    done
done

echo "All experiments completed!"
echo "Results saved in ./experiments/logs/"
echo "To analyze results, check the individual experiment directories."