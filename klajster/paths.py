import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(__file__)).parent.absolute()
PROJECT_DIR = ROOT_DIR / "klajster"

DATA_PATH = ROOT_DIR / "data"
DATASETS_PATH = DATA_PATH / "datasets"

# CONFIGS
EXPERIMENTS_PATH = ROOT_DIR / "experiments"
CONFIGS_PATH = EXPERIMENTS_PATH / "configs"
DATASETS_CFG_PATH = CONFIGS_PATH / "datasets"
HPS_COMMON_CFG_PATH = CONFIGS_PATH / "hps" / "common.yaml"

# OUTPUT PATHS
# HPS
HPS_OUTPUT_PATH = DATA_PATH / "hps"
LIGHTNING_HPS_OUTPUT_PATH = HPS_OUTPUT_PATH / "lightning"

# PIPELINES
MODELS_OUTPUT_PATH = DATA_PATH / "models"
LIGHTNING_PIPELINE_OUTPUT_PATH = MODELS_OUTPUT_PATH / "lightning"
PIPELINES_OUTPUT_PATHS_MAPPING = {
    "lightning": LIGHTNING_PIPELINE_OUTPUT_PATH,
}


def get_dataset_config_path(dataset_name: str) -> Path:
    return DATASETS_CFG_PATH / f"{dataset_name}.yaml"


def build_embedding_path_by_name(embedding_name: str) -> str:
    return embedding_name.replace("/", "__")


def get_embedding_name_by_path(embedding_path: str) -> str:
    return embedding_path.replace("__", "/")
