#!/bin/bash
# Launch scanner-transfer training for all prefeatures configs.
# Usage: bash src/histaug/scripts/train_all_prefeatures.sh [--parallel]
#   --parallel  Run all jobs simultaneously (round-robin GPU assignment).
#               Default: sequential.

set -euo pipefail

PARALLEL=false
for arg in "$@"; do
    case "$arg" in
        --parallel) PARALLEL=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAIN_DIR="$REPO_ROOT/src/histaug"

NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}

configs=(
    # src+tgt scanner conditioning
    # "config/Histaug_phikon_prefeatures.yaml"
    "config/Histaug_h0mini_prefeatures.yaml"
    # tgt_scanner-only conditioning
    # "config/Histaug_phikon_prefeatures_tgt.yaml"
    "config/Histaug_h0mini_prefeatures_tgt.yaml"
)

if $PARALLEL; then
    echo "Parallel mode: $NUM_GPUS GPU(s) detected, launching ${#configs[@]} jobs."
    pids=()
    for i in "${!configs[@]}"; do
        cfg="${configs[$i]}"
        gpu=$(( i % NUM_GPUS ))
        logfile="$TRAIN_DIR/logs/train_$(basename "$cfg" .yaml).log"
        echo "Launching [$cfg] on GPU $gpu → $logfile"
        CUDA_VISIBLE_DEVICES=$gpu python "$TRAIN_DIR/train.py" \
            --stage=train --config "$cfg" \
            > "$logfile" 2>&1 &
        pids+=($!)
    done

    echo "All jobs launched (PIDs: ${pids[*]}). Waiting..."
    failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "DONE:   ${configs[$i]}"
        else
            echo "FAILED: ${configs[$i]}"
            failed=$(( failed + 1 ))
        fi
    done
else
    echo "Sequential mode: running ${#configs[@]} jobs on GPU 0."
    failed=0
    for cfg in "${configs[@]}"; do
        echo "========================================"
        echo "Training: $cfg"
        echo "========================================"
        if ! (cd "$TRAIN_DIR" && python train.py --stage=train --config "$cfg"); then
            echo "FAILED: $cfg"
            failed=$(( failed + 1 ))
        fi
    done
fi

echo "Finished. $failed job(s) failed."
exit $failed
