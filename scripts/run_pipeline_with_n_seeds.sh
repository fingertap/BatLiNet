#!/bin/bash
SCRIPT=./scripts/pipeline.py
CONFIG_ROOT="$(realpath configs)"
CONFIG="$(realpath $1)"
WORKSPACE_ROOT=./workspaces

# Total number of seeds
TOTAL_SEEDS=$2

# Check if nvidia-smi is available and count the number of GPUs
USE_GPU=1
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    echo "No GPUs found, using CPU."
    NUM_GPUS=1
    USE_GPU=0
fi

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
relpath=$(echo $CONFIG | sed "s|$CONFIG_ROOT/||")
WORKSPACE=$WORKSPACE_ROOT/${relpath%.*}

mkdir -p $WORKSPACE

export PYTHONPATH=$PYTHONPATH:.

echo Processing $relpath

# Initialize an array to keep track of running processes on each device
declare -A device_pids

for (( seed=0; seed<TOTAL_SEEDS; seed++ )); do
    device_id=$((seed % NUM_GPUS))
    if [ $USE_GPU -eq 1 ]; then
        actual_device="cuda:$device_id"
    else
        actual_device="cpu"
    fi

    # Wait for the current device to be free before starting a new job
    if [[ -n ${device_pids[$device_id]} ]]; then
        wait ${device_pids[$device_id]}
    fi

    [[ -e $WORKSPACE/log.$seed ]] || python $SCRIPT $1 --train --device $actual_device --workspace $WORKSPACE --seed $seed | tee $WORKSPACE/log.$seed &
    device_pids[$device_id]=$!
done

# wait for all remaining pids
for pid in ${device_pids[@]}; do
    wait $pid
done
