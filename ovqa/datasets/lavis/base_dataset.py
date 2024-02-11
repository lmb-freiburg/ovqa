"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import logging
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from typing import Iterable

from ovqa.utils_for_ext import make_list_smaller

logger = logging


class BaseDataset(Dataset):
    def __init__(
        self,
        vis_processor=None,
        text_processor=None,
        vis_root=None,
        ann_paths=None,
        config=None,
    ):
        self.config = config
        self.vis_root = vis_root

        if ann_paths is None:
            ann_paths = []

        self.annotation = []
        for ann_path in ann_paths:
            print(f"Loading annotation from {ann_path}")
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def update_from_config(self):
        """
        Will be called after init by dataset builder to modify the dataset based on its config.
        """
        # reduce dataset size on the fly
        debug_max = self.config.get("debug_max", None)
        debug_start = self.config.get("debug_start", 0)
        self.annotation = make_list_smaller(self.annotation, debug_start, debug_max)
        logger.warning(
            f"Final dataset size of {type(self).__name__}: {len(self.annotation)} samples. "
            f"Start index {debug_start} max {debug_max}"
        )
        if hasattr(self, "image"):
            # for classifier retrieval dataset
            self.image = make_list_smaller(self.image, debug_start, debug_max)

        # selected datapoints for brittleness experiments
        if self.config.get("selected_datapoints", "") != "":
            # check file exists
            assert Path(
                self.config["selected_datapoints"]
            ).exists(), "File {} does not exist".format(self.config["selected_datapoints"])
            selected_datapoints = json.load(open(self.config["selected_datapoints"], "r"))
            selected_datapoints.sort()
            self.annotation = [self.annotation[i] for i in selected_datapoints]
            logger.warning(
                f"Final dataset size of {type(self).__name__}: {len(self.annotation)} samples. "
                f"Selected datapoints from {self.config['selected_datapoints']}"
            )
            if hasattr(self, "image"):
                # for classifier retrieval dataset
                self.image = [self.image[i] for i in selected_datapoints]

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class LavisConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # for now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)  # noqa
