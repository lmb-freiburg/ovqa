"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging

from ovqa.common.lavis.registry import registry
from ovqa.tasks.base_task import BaseTask
from ovqa.tasks.multimodal_classification import MultimodalClassificationTask
from ovqa.tasks.vqa import VQATask, GQATask, AOKVQATask
from ovqa.tasks.classifier_vqa_followup_task import ClassifierVQAFollowupTask
from ovqa.tasks.classifier_vqa_task import ClassifierVQATask
from ovqa.tasks.multilabel_classifier_vqa_task import MultilabelClassifierVQATask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    logging.info(f"Setting up task: {task_name}")
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "AOKVQATask",
    "VQATask",
    "GQATask",
    "MultimodalClassificationTask",
    "ClassifierVQATask",
    "ClassifierVQAFollowupTask",
    "MultilabelClassifierVQATask",
]
