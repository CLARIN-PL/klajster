export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH="/myrun/runs/miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

if [ $SLURM_LOCALID -eq 0 ] ; then
  rm -rf $MIOPEN_USER_DB_PATH
  mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 3
