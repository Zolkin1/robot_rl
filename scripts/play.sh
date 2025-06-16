#!/bin/bash
export CUDA_VISIBLE_DEVICES=1  # or 1,2,...
# === Configurable Inputs ===
env_type="clf"
num_env=2
experiment_name="g1"
sim_speed="1.0,0,0"

# Construct the root log directory
log_dir="logs/g1_policies/${env_type}/${experiment_name}"
play_log_root_dir="intermediate_log/${env_type}"
mkdir -p "$play_log_root_dir"

# === Find the latest run directory ===
latest_run=$(find "$log_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$latest_run" ]; then
    echo "No run directory found in $log_dir"
    exit 1
fi

# === Find all model_*.pt checkpoints and sort them ===
checkpoints=$(find "$latest_run" -name 'model_*.pt' | sort)

if [ -z "$checkpoints" ]; then
    echo "No checkpoint files found in $latest_run"
    exit 1
fi


# === Iterate through each checkpoint ===
for ckpt_path in $checkpoints; do
    # Extract checkpoint number (e.g., model_123.pt → 123)
    step=$(basename "$ckpt_path" | sed -E 's/model_([0-9]+)\.pt/\1/')

    # Create subfolder for this checkpoint's outputs
    play_dir="${play_log_root_dir}/play_step_${step}"
    mkdir -p "$play_dir"

    echo "Running checkpoint ${ckpt_path}..."

    python scripts/rsl_rl/play_policy.py \
        --env_type="${env_type}" \
        --headless \
        --num_env="${num_env}" \
        --sim_speed="${sim_speed}" \
        --play_log_dir="${play_dir}" \
        --checkpoint="model_${step}.pt"  # full path passed here
done

