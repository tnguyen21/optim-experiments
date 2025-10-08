#!/bin/bash

# Multi-seed experiment script with optimized configurations
# Now supports scale combinations: optimizer x scale x seed
# Usage: ./scripts/run_multi_seed_experiment.sh [scale1 scale2 ...]
# Example: ./scripts/run_multi_seed_experiment.sh tiny small
#          ./scripts/run_multi_seed_experiment.sh (defaults to tiny only)

set -e  # Exit on any error

# Configuration
SEEDS=(42 123 456 314 2718)

# Parse command line arguments for scales
if [ $# -eq 0 ]; then
    # Default to tiny scale only for backward compatibility
    SCALES=("tiny")
else
    # Use provided scales
    SCALES=("$@")
fi

# Scale-specific optimizer configurations
declare -A SCALE_OPTIMIZER_CONFIGS
# Tiny scale configs
SCALE_OPTIMIZER_CONFIGS["tiny_sgd"]="config/tiny_sgd_optimal.yaml"
SCALE_OPTIMIZER_CONFIGS["tiny_adamw"]="config/tiny_adamw_optimal.yaml"
SCALE_OPTIMIZER_CONFIGS["tiny_muon"]="config/tiny_muon_optimal.yaml"
# Small scale configs
SCALE_OPTIMIZER_CONFIGS["small_sgd"]="config/small_sgd_optimal.yaml"
SCALE_OPTIMIZER_CONFIGS["small_adamw"]="config/small_adamw_optimal.yaml"
SCALE_OPTIMIZER_CONFIGS["small_muon"]="config/small_muon_optimal.yaml"
# Medium scale configs
SCALE_OPTIMIZER_CONFIGS["medium_sgd"]="config/medium_sgd_optimal.yaml"
SCALE_OPTIMIZER_CONFIGS["medium_adamw"]="config/medium_adamw_optimal.yaml"
SCALE_OPTIMIZER_CONFIGS["medium_muon"]="config/medium_muon_optimal.yaml"

OPTIMIZERS=("sgd" "adamw" "muon")

echo "Starting multi-scale multi-seed optimizer comparison experiment..."
echo "Scales: ${SCALES[@]}"
echo "Optimizers: ${OPTIMIZERS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo
echo "Scale-optimizer configs:"
for scale in "${SCALES[@]}"; do
    for opt in "${OPTIMIZERS[@]}"; do
        config_key="${scale}_${opt}"
        config_file="${SCALE_OPTIMIZER_CONFIGS[$config_key]}"
        if [ -f "$config_file" ]; then
            echo "  $scale + $opt: $config_file"
        else
            echo "  $scale + $opt: CONFIG NOT FOUND ($config_file)"
        fi
    done
done
echo

# Function to run a single experiment
run_experiment() {
    local scale=$1
    local optimizer=$2
    local seed=$3
    local config_key="${scale}_${optimizer}"
    local config_file="${SCALE_OPTIMIZER_CONFIGS[$config_key]}"
    
    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config file not found: $config_file"
        echo "Skipping experiment: scale=$scale, optimizer=$optimizer, seed=$seed"
        return 1
    fi
    
    echo "Running experiment: scale=$scale, optimizer=$optimizer, seed=$seed"
    echo "Config: $config_file"
    
    python scripts/run_experiment.py \
        --config "$config_file" \
        --seed $seed
    
    echo "Completed: scale=$scale, optimizer=$optimizer, seed=$seed"
    echo "----------------------------------------"
}

# Run all combinations
total_experiments=$((${#SCALES[@]} * ${#OPTIMIZERS[@]} * ${#SEEDS[@]}))
current_experiment=0

for scale in "${SCALES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            current_experiment=$((current_experiment + 1))
            echo "Experiment $current_experiment/$total_experiments"
            run_experiment $scale $optimizer $seed
        done
    done
done

echo "All experiments completed!"
echo "Results saved in ./experiments/logs/"
echo "To analyze results, check the individual experiment directories."