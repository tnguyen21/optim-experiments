#!/bin/bash

# Model Scale Sweep Script
# Runs experiments across multiple model scales with specified optimizer(s) and seed(s)
# Usage: ./scripts/run_scale_sweep.sh [OPTIMIZER] [SEED]
# Examples:
#   ./scripts/run_scale_sweep.sh                    # All optimizers, seed 42, all scales
#   ./scripts/run_scale_sweep.sh muon               # Only Muon, seed 42, all scales
#   ./scripts/run_scale_sweep.sh adamw 123          # Only AdamW, seed 123, all scales

set -e  # Exit on any error

# Configuration
DEFAULT_SCALES=("tiny" "small" "medium")
ALL_OPTIMIZERS=("sgd" "adamw" "muon")
DEFAULT_SEED=42

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments: run all optimizers with default seed
    OPTIMIZERS=("${ALL_OPTIMIZERS[@]}")
    SEED=$DEFAULT_SEED
elif [ $# -eq 1 ]; then
    # One argument: specific optimizer with default seed
    OPTIMIZERS=("$1")
    SEED=$DEFAULT_SEED
elif [ $# -eq 2 ]; then
    # Two arguments: specific optimizer and seed
    OPTIMIZERS=("$1")
    SEED=$2
else
    echo "Usage: $0 [OPTIMIZER] [SEED]"
    echo "  OPTIMIZER: sgd, adamw, muon (default: all)"
    echo "  SEED: random seed (default: 42)"
    exit 1
fi

SCALES=("${DEFAULT_SCALES[@]}")

# Validate optimizer
for opt in "${OPTIMIZERS[@]}"; do
    if [[ ! " ${ALL_OPTIMIZERS[@]} " =~ " ${opt} " ]]; then
        echo "Error: Invalid optimizer '$opt'. Must be one of: ${ALL_OPTIMIZERS[*]}"
        exit 1
    fi
done

echo "Starting Model Scale Sweep..."
echo "Scales: ${SCALES[@]}"
echo "Optimizers: ${OPTIMIZERS[@]}"
echo "Seed: $SEED"
echo

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

# Check that all required config files exist
echo "Checking config files:"
all_configs_exist=true
for scale in "${SCALES[@]}"; do
    for opt in "${OPTIMIZERS[@]}"; do
        config_key="${scale}_${opt}"
        config_file="${SCALE_OPTIMIZER_CONFIGS[$config_key]}"
        if [ -f "$config_file" ]; then
            echo "  ✓ $scale + $opt: $config_file"
        else
            echo "  ✗ $scale + $opt: CONFIG NOT FOUND ($config_file)"
            all_configs_exist=false
        fi
    done
done

if [ "$all_configs_exist" = false ]; then
    echo
    echo "Error: Some config files are missing. Please create them first."
    exit 1
fi

echo

# Function to run a single experiment
run_experiment() {
    local scale=$1
    local optimizer=$2
    local config_key="${scale}_${optimizer}"
    local config_file="${SCALE_OPTIMIZER_CONFIGS[$config_key]}"
    
    echo "Running experiment: scale=$scale, optimizer=$optimizer, seed=$SEED"
    echo "Config: $config_file"
    
    python scripts/run_experiment.py \
        --config "$config_file" \
        --seed $SEED
    
    echo "Completed: scale=$scale, optimizer=$optimizer, seed=$SEED"
    echo "----------------------------------------"
}

# Calculate total experiments
total_experiments=$((${#SCALES[@]} * ${#OPTIMIZERS[@]}))
current_experiment=0

echo "Running $total_experiments experiments..."
echo

# Run experiments for each scale and optimizer combination
results_summary=()
for scale in "${SCALES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
        current_experiment=$((current_experiment + 1))
        echo "Experiment $current_experiment/$total_experiments"
        
        start_time=$(date +%s)
        if run_experiment "$scale" "$optimizer"; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            results_summary+=("$scale + $optimizer: SUCCESS (${duration}s)")
        else
            results_summary+=("$scale + $optimizer: FAILED")
        fi
        echo
    done
done

echo "========================================="
echo "SCALE SWEEP COMPLETED"
echo "========================================="
echo "Summary:"
for result in "${results_summary[@]}"; do
    echo "  $result"
done
echo
echo "Results saved in ./experiments/logs/"
echo "To analyze results across scales, run:"
echo "  python scripts/analyze_results.py --scale-comparison"