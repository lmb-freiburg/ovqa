"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import shutil
import torch.distributed as dist
import warnings
from omegaconf import OmegaConf
from pathlib import Path
from torchvision.datasets.utils import download_url

import ovqa.common.lavis.utils as utils
from ovqa.common.lavis.dist_utils import is_dist_avail_and_initialized, is_main_process
from ovqa.common.lavis.registry import registry
from ovqa.paths import get_ovqa_cache_dir
from ovqa.processors.base_processor import BaseProcessor
from packg import format_exception


class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            # help to create datasets from default config.
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.data_type = self.config.data_type

        self.vis_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        if is_main_process():
            self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build_processors(self):
        vis_proc_cfg = self.config.get("vis_processor")
        txt_proc_cfg = self.config.get("text_processor")

        if vis_proc_cfg is not None:
            vis_train_cfg = vis_proc_cfg.get("train")
            vis_eval_cfg = vis_proc_cfg.get("eval")

            self.vis_processors["train"] = self._build_proc_from_cfg(vis_train_cfg)
            self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")

            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)

    @staticmethod
    def _build_proc_from_cfg(cfg):
        if cfg is None:
            return None

        proc_cls = registry.get_processor_class(cfg.name)
        if proc_cls is None:
            raise ValueError(f"Unknown processor type {cfg.name}")

        return proc_cls.from_config(cfg)

    @classmethod
    def default_config_path(cls, type="default"):
        return utils.get_abs_path(cls.DATASET_CONFIG_DICT[type])

    def _download_data(self):
        self._download_ann()
        self._download_vis()

    def _download_ann(self):
        """
        Download annotation files if necessary.
        All the vision-language datasets should have annotations of unified format.

        storage_path can be:
          (1) relative/absolute: will be prefixed with env.cache_root to make full path if relative.
          (2) basename/dirname: will be suffixed with base name of URL if dirname is provided.

        Local annotation paths should be relative.
        """
        anns = self.config.build_info.annotations

        splits = anns.keys()

        cache_root = get_ovqa_cache_dir()

        for split in splits:
            info = anns[split]
            urls, storage_paths = info.get("url", None), info.storage

            if isinstance(urls, str):
                urls = [urls]
            if isinstance(storage_paths, str):
                storage_paths = [storage_paths]

            if urls is None:
                return

            assert len(urls) == len(storage_paths)

            for url_or_filename, storage_path in zip(urls, storage_paths):
                storage_path = Path(storage_path)
                if storage_path.is_file():
                    logging.info("Using existing file {}.".format(storage_path.as_posix()))
                    continue
                if not storage_path.is_absolute():
                    storage_path = (cache_root / storage_path).resolve().absolute()

                if storage_path.is_dir():
                    raise ValueError(
                        f"Expecting storage_path to be a file path, got directory {storage_path}"
                    )

                dirname = storage_path.parent
                os.makedirs(dirname, exist_ok=True)

                if os.path.isfile(url_or_filename):
                    if not storage_path.is_file():
                        shutil.copyfile(src=url_or_filename, dst=storage_path.as_posix())
                        continue
                    logging.info("Using existing file {}.".format(storage_path.as_posix()))
                    continue
                filename = storage_path.name
                try:
                    download_url(url=url_or_filename, root=dirname.as_posix(), filename=filename)
                except ValueError as e:
                    logging.warning(
                        f"Cannot find or download {url_or_filename} - {format_exception(e)}"
                    )

    def _download_vis(self):
        storage_path = self.config.build_info.get(self.data_type).storage
        if not os.path.exists(storage_path):
            logging.error(
                f"""
                ******************************************************************\n
                The specified path {storage_path} for visual inputs does not exist.
                Please provide a correct path to the visual inputs or
                refer to DATASETS.md for downloading instructions.
                ******************************************************************\n
                """
            )

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            # # change: removed this to allow custom splits
            # if split not in ["train", "val", "test"]:
            #     continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"] if is_train else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"] if is_train else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            # check if ann_paths are real paths or relative to cache dir
            new_ann_paths = []
            for ann_path in ann_paths:
                ann_path = Path(ann_path)
                if not ann_path.is_file():
                    ann_path_new = get_ovqa_cache_dir() / ann_path
                    if not ann_path_new.is_file():
                        raise FileNotFoundError(
                            f"Could not find either {ann_path.as_posix()} or {ann_path_new.as_posix()}"
                        )
                    ann_path = ann_path_new
                new_ann_paths.append(ann_path)
            ann_paths = new_ann_paths

            # visual data storage path
            vis_path = vis_info.storage
            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            self.config.build_info.annotations[split]["split"] = split

            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                config=self.config,
            )

            datasets[split].update_from_config()  # custom add config to the dataset

        return datasets


def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    cfg = cfg[list(cfg.keys())[0]]

    return cfg
