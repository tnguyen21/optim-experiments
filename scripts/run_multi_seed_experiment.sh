#!/bin/bash

# Multi-seed experiment script for comparing optimizers
# Usage: ./scripts/run_multi_seed_experiment.sh

set -e  # Exit on any error

# Configuration
SEEDS=(42 123 456)
OPTIMIZERS=("sgd" "adamw" "muon")
CONFIG_FILE="config/base_config.yaml"

echo "Starting multi-seed optimizer comparison experiment..."
echo "Seeds: ${SEEDS[@]}"
echo "Optimizers: ${OPTIMIZERS[@]}"
echo "Config: $CONFIG_FILE"
echo

# Function to run a single experiment
run_experiment() {
    local optimizer=$1
    local seed=$2
    
    echo "Running experiment: optimizer=$optimizer, seed=$seed"
    python scripts/run_experiment.py \
        --config $CONFIG_FILE \
        --optimizer $optimizer \
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