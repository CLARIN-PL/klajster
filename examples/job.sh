#!/bin/bash
$WITH_CONDA
GPUS=$1
NUM_NODES=$2
MAX_GPU_ID=$((GPUS-1))

export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH="/myrun/runs/miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

if [ $SLURM_LOCALID -eq 0 ] ; then
  rm -rf $MIOPEN_USER_DB_PATH
  mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 3

echo "CUSTOM CACHE DIR: $MIOPEN_CUSTOM_CACHE_DIR"
echo "USER DB PATH: $MIOPEN_USER_DB_PATH"
echo "GPUS: $GPUS"

python /myrun/examples/torch/train_lightning.py \
    --rootdir /myrun/runs/torch-example  \
    --gpu_device $(seq -s ' ' 0 $MAX_GPU_ID) \
    --batch_size 768 --num_nodes $NUM_NODES
