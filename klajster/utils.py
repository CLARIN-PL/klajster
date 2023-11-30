import importlib
from pathlib import Path
from typing import Any, Sequence, Tuple, Type, Union

import yaml
from datasets import disable_caching
from embeddings.utils.loggers import LightningLoggingConfig
from yaml.loader import Loader


def read_yaml(filepath: Union[str, Path], safe_load: bool = True) -> Any:
    with open(filepath, "r") as f:
        if safe_load:
            return yaml.safe_load(f)
        else:
            return yaml.load(f, Loader=Loader)


def get_module_from_str(module: str) -> Type[Any]:
    module, cls = module.rsplit(".", maxsplit=1)
    cls = getattr(importlib.import_module(module), cls)
    return cls  # type: ignore


def get_create_eval_paths(
    results_path: Path, models_path: Path, dataset_name: str, embedding_name: str
) -> Tuple[str, str]:
    persist_out_path = results_path.joinpath(dataset_name, f"{embedding_name}.json")
    persist_out_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = models_path.joinpath(dataset_name, embedding_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(persist_out_path), str(output_path)


def parse_dataset_cfg_for_evaluation(
    dataset_cfg_path: str, cfg_type: str, gpu_type: str
) -> Tuple[str, str, Union[str, Sequence[str]], str]:
    ds_cfg = read_yaml(dataset_cfg_path)
    return (
        ds_cfg["name"],
        ds_cfg["paths"][cfg_type][gpu_type],
        ds_cfg["common_args"]["input_column_names"],
        ds_cfg["common_args"]["target_column_name"],
    )


def disable_hf_datasets_caching() -> None:
    # Disable generation of datasets cache files
    disable_caching()  # type: ignore


def get_lightning_logging_config(
    tracking_project_name: str = "klajster",
    wandb_entity: str = "graph-ml-lab-wust",
    wandb_run_id: str | None = None,
) -> LightningLoggingConfig:
    return LightningLoggingConfig(
        loggers_names=["wandb"],
        tracking_project_name=tracking_project_name,
        wandb_entity=wandb_entity,
        wandb_run_id=wandb_run_id,
    )
