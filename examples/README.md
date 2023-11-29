based on: https://github.com/dnddnjs/pytorch-multigpu/tree/master


# Example command 

```
srun --partition=<partition> --account=<project_name> singularity exec -B /scratch/<project_name> ubuntu_21.04.sif ls /scratch/<account>

<project_name> = project_465000858
<partition> = dev-g | standard-g
```

dev-g -> pojedyncze gpu
standard-g -> całe węzły na wyłączność



#srun -q tesla -p tesla --mem=32gb -n 1 -I --gres=gpu:1 -t 03:00:00 --cpus-per-gpu=12 --pty bash
#singularity -B shell klajster.sif


