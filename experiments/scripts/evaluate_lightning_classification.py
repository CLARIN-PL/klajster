import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Union

import typer
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from tqdm.auto import tqdm

from klajster.paths import LIGHTNING_HPS_OUTPUT_PATH, get_dataset_config_path
from klajster.utils import (
    disable_hf_datasets_caching,
    get_lightning_logging_config,
    parse_dataset_cfg_for_evaluation,
    read_yaml,
)

app = typer.Typer()


def get_task_run_name(embedding_path: str, dataset: str, devices: int, run_id: int) -> str:
    return f"{dataset}_{embedding_path}_devices_{devices}_run_{run_id}"


def get_model_checkpoint_kwargs() -> Dict[str, Union[str, None, bool]]:
    return {
        "filename": "last",
        "monitor": None,
        "save_last": False,
    }


def write_evaluation_results(output_path: Union[str, Path], elapsed_time: float):
    with open(output_path / "evaluation.json", "r") as f:
        evaluation_results = json.load(f)
    del evaluation_results["data"]
    evaluation_results = {"total_time_in_sec": elapsed_time, "metrics": evaluation_results}
    with open(output_path / "evaluation_lumi.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)


def run(
    embedding_path: str = typer.Option("...", help="Embedding path."),
    ds: str = typer.Option("...", help="Dataset name."),
    pipeline_params_path: Path = typer.Option(
        "...",
        help="Path to pipeline parameters config. If `None` the script will try to read the parameters configuration from a default HPS location",
    ),
    output_path: Path = typer.Option("...", help="Output path for model results."),
    num_nodes: int = typer.Option("...", help="Number of compute nodes."),
    devices: int = typer.Option("...", help="Number of GPUs."),
    accelerator: str = typer.Option("...", help="Accelerator type."),
    retrains: int = typer.Option(1, help="Number of model retrains."),
    wandb_entity: str = typer.Option("graph-ml-lab-wust", help="WandB entity."),
    tracking_project_name: str = typer.Option("klajster", help="WandB project name."),
) -> None:
    disable_hf_datasets_caching()
    output_path.mkdir(exist_ok=True, parents=True)

    (
        dataset_name,
        dataset_path,
        input_column_name,
        target_column_name,
    ) = parse_dataset_cfg_for_evaluation(str(get_dataset_config_path(ds)), cfg_type="lightning")

    if pipeline_params_path is None:  # HPS
        pipeline_params_path = LIGHTNING_HPS_OUTPUT_PATH / embedding_path / ds / "best_params.yaml"
        assert pipeline_params_path.is_file()

    cfg = read_yaml(pipeline_params_path, safe_load=False)
    cfg["dataset_name_or_path"] = dataset_path
    cfg["output_path"] = output_path
    cfg["logging_config"] = get_lightning_logging_config(
        tracking_project_name=tracking_project_name, wandb_entity=wandb_entity
    )
    cfg["model_checkpoint_kwargs"] = get_model_checkpoint_kwargs()
    cfg["devices"] = devices
    cfg["accelerator"] = accelerator
    cfg["config"].task_train_kwargs["num_nodes"] = num_nodes

    for run_id in tqdm(range(retrains), desc="Run"):
        run_cfg = deepcopy(cfg)
        run_output_path = output_path / f"run-{run_id}"
        run_output_path.mkdir(parents=False, exist_ok=True)
        run_cfg["output_path"] = run_output_path
        pipeline = LightningClassificationPipeline(**run_cfg)
        run_name = get_task_run_name(
            dataset=dataset_name, embedding_path=embedding_path, devices=devices, run_id=run_id
        )
        start = time.time()
        pipeline.run(run_name=run_name)
        end = time.time()
        write_evaluation_results(output_path=run_output_path, elapsed_time=(end - start))


if __name__ == "__main__":
    typer.run(run)
