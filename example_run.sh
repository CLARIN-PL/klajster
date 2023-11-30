srun \
  -t 0:10:0 \
  -N $NODES \
  --ntasks-per-node $GPUS_PER_NODE \
  --partition=standard-g \
  --account=project_465000858 \
  --gpus $((NODES*GPUS_PER_NODE)) \
  singularity exec \
    -B $(pwd):/myrun \
    /storage/project_465000858/klajster.sif \
    /myrun/examples/torch/job.sh $GPUS_PER_NODE $NODES
