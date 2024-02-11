"""
Example

# 1) setup dataset
dataset_name = "imagenet1k"
dataset_split = "val"
meta: ClsMetadataInterface = meta_loader.load_metadata(dataset_name, dataset_split)
classes_data: List[Dict[str, Any]] = meta.classes_data
class_names: List[str] = meta.get_class_list()
targets: Dict[str, int] = meta.get_targets()
ann: Dict[str, Dict[str, Any]] = meta.get_annotations()

# 2) Load clip
clip_model_name, clip_model_pretrained = "EVA01-g-14", "laion400m_s11b_b41k"
# clip_model_name, clip_model_pretrained = "ViT-L-14", "openai"
# clip_model_name, clip_model_pretrained = "EVA02-E-14-plus", "laion2b_s9b_b144k"

clip_acc, clip_logits = load_clip_result(
    dataset_name, dataset_split, clip_model_name, clip_model_pretrained)

# clip_acc, clip_logits = load_clip_result_from_dir(
#     "/home/gings/repos/others/CLIP_benchmark/output/imagenet1k-square~test~openai~ViT-L-14~en~zeroshot_classification~auto~default/")
print(f"Clip acc: {clip_acc.mean().item():.1%}")

"""
from pathlib import Path
from typing import Dict

import torch

from packg.caching import get_joblib_memory
from packg.iotools import load_json
from ovqa.paths import get_ovqa_annotations_dir
from ovqa.paths import get_ovqa_output_dir
from ovqa.metrics.torchmetrics_ext import MetricExt

mem = get_joblib_memory()


def convert_text_preds_to_logits(preds: Dict[str, str], targets: Dict[str, str], metric: MetricExt):
    pred_keys = list(preds.keys())
    pred_values = list(preds.values())
    target_ids = [targets[t_key] for t_key in pred_keys]
    metric.reset()
    metric.update(pred_keys, pred_values, target_ids)
    t_logits = metric.compute_logits()
    pred_ids = t_logits.argmax(-1)
    t_acc_tensor = (torch.tensor(target_ids) == pred_ids).float()
    return (
        t_acc_tensor,  # shape (n_preds,)
        t_logits,  # shape (n_preds, n_classes)
    )


def load_clip_result(
    t_targets,
    t_dataset,
    t_split,
    t_model,
    t_pretrained,
    t_prompt="openai_imagenet_template",
    clip_result_dir=get_ovqa_output_dir() / "clip_benchmark",
):
    t_clip_dataset = {"caltech101": "vic_caltech101"}.get(t_dataset, t_dataset)
    # imagenet1k-square
    t_image_preprocessing = "default"

    t_clip_name = (
        f"{t_clip_dataset}~{t_split}~{t_pretrained}~{t_model}~"
        f"en~zeroshot_classification~{t_prompt}~"
        f"{t_image_preprocessing}"
    )
    return load_clip_result_from_dir(
        t_targets, clip_result_dir / t_dataset / t_clip_name, t_dataset
    )


def load_clip_result_from_dir(t_targets, t_dir, t_dataset_name="any"):
    t_dir = Path(t_dir)
    t_num = load_json(t_dir / "result.json")
    print(f"Loading {t_dir.name} with result {t_num}")

    t_logits = torch.load(t_dir / "result_logits.pt")
    if t_dataset_name.startswith("imagenet1k"):
        # clip benchmark sorts the imagenet dataset differently so we have to resort the logits
        map_id_to_image_file = (
            get_ovqa_annotations_dir() / "imagenet1k" / "filemap" / "map_id_to_imagefile-val.json"
        )
        file_map = load_json(map_id_to_image_file)
        default_sort = {i: v for i, v in enumerate(file_map.values())}
        new_sort = [i for i, _ in sorted(default_sort.items(), key=lambda x: x[1])]
        new_sort = torch.tensor(new_sort)

        # resort logits - clip datapoint 0 will be sorted as my datapoint new_sort[0]
        t_logits[new_sort] = t_logits.clone()

    # targets sorted by my order
    t_target_ids = torch.tensor(list(t_targets.values()))

    # # targets sorted by clip order
    # t_target_ids = torch.load(t_dir / "result_target.pt")
    t_pred_ids = t_logits.argmax(-1)
    t_acc_tensor = (t_target_ids == t_pred_ids).float()
    return t_acc_tensor, t_logits
