import io

from ovqa import tasks
from ovqa.common.lavis.config import Config
from ovqa.datasets.lavis.vqav2_datasets import VQAv2EvalDataset
from ovqa.tasks import VQATask
from packg.iotools import dumps_yaml


def load_lavis_dataset(dataset_name, dataset_split, task, dataset_type="eval"):
    """
    Build config to create datasets only.
    Note: model and most of the run config are never used, but needed to create the task
    """

    config = {
        "datasets": {dataset_name: {"type": dataset_type}},
        "run": {
            "task": task,
            "test_splits": [dataset_split],
            "evaluate": True,
            "distributed": False,
            "num_beams": None,
            "max_new_tokens": None,
            "min_new_tokens": None,
            "prompt": None,
            "inference_method": None,
        },
        "model": {
            "arch": "blip2_t5",
            "model_type": "pretrain_flant5xl",
        },
    }
    config_stream = io.StringIO(dumps_yaml(config))
    cfg = Config.from_file(config_stream)
    task: VQATask = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    dataset: VQAv2EvalDataset = datasets[dataset_name][dataset_split]
    return dataset
