#!/bin/bash
source .env
SCRIPT=$CODE_HOME/scripts/pipeline.py
CONFIG_ROOT="$(realpath configs)"
CONFIG="$(realpath $1)"

DEVICES=8
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
relpath=$(echo $CONFIG | sed "s|$CONFIG_ROOT/||")
WORKSPACE=$WORKSPACE_ROOT/${relpath%.*}

mkdir -p $WORKSPACE

export PYTHONPATH=$PYTHONPATH:$CODE_HOME

echo Processing $relpath

for device in `seq 0 $((DEVICES-1))`; do
    [[ -e $WORKSPACE/log.$device ]] || python $SCRIPT $1 --train --device cuda:$device --workspace $WORKSPACE --seed $device | tee $WORKSPACE/log.$device &
    pids[${device}]=$!
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done
