# KLAJSTER

### How to prepare env?
See: `https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/`

```bash
# To create env
module load LUMI
module load lumi-container-wrapper

pip-containerize new --prefix .venv_container/ requirements.txt

# To use env
export PATH="/flash/project_465000858/klajster/.venv_container/bin:$PATH"
```
