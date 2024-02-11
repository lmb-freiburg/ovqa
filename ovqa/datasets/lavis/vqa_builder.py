"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from pathlib import Path

from ovqa.common.lavis.registry import registry
from ovqa.datasets.create_vqav2_subsplit import create_vqav2_subsplit
from ovqa.datasets.lavis.base_dataset_builder import BaseDatasetBuilder
from ovqa.datasets.lavis.gqa_datasets import GQADataset, GQAEvalDataset
from ovqa.datasets.lavis.vqav2_datasets import VQAv2Dataset, VQAv2EvalDataset
from ovqa.paths import get_ovqa_repo_root


@registry.register_builder("vqav2")
class VQAv2Builder(BaseDatasetBuilder):
    train_dataset_cls = VQAv2Dataset
    eval_dataset_cls = VQAv2EvalDataset

    DATASET_CONFIG_DICT = {
        "eval": "ovqa/configs/lavis_datasets/vqav2/eval.yaml",
    }

    def build(self):
        vqav2_ann_dir = get_ovqa_repo_root() / "ovqa/annotations/vqav2"
        create_vqav2_subsplit(vqav2_ann_dir / "minival_ids.json")
        create_vqav2_subsplit(vqav2_ann_dir / "nominival_ids.json")
        return super().build()


@registry.register_builder("gqa")
class GQABuilder(BaseDatasetBuilder):
    train_dataset_cls = GQADataset
    eval_dataset_cls = GQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "ovqa/configs/lavis_datasets/gqa/defaults.yaml",
        "balanced_val": "ovqa/configs/lavis_datasets/gqa/balanced_val.yaml",
        "balanced_testdev": "ovqa/configs/lavis_datasets/gqa/balanced_testdev.yaml",
        "eval": "ovqa/configs/lavis_datasets/gqa/eval.yaml",
    }
    # created "eval" with splits
    # "balanced_train", "balanced_val", "balanced_test", "balanced_testdev"
    # from those, test does not have annotations.
    # balanced_testdev is the best, because also instructblip did not train on it.
