#!/bin/bash
$WITH_CONDA
GPUS=$1
NUM_NODES=$2
echo "GPUS: $GPUS"
MAX_GPU_ID=$((GPUS-1))
python /myrun/examples/torch/train_lightning.py --rootdir /myrun/runs/torch-example --gpu_device $(seq -s ' ' 0 $MAX_GPU_ID) --batch_size 768 --num_nodes $NUM_NODES
