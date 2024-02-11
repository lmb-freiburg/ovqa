"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import hashlib
import itertools
import json
import logging
import os
import shutil
import torch
from pathlib import Path
from torchmetrics import Accuracy

from ovqa.common.lavis.dist_utils import main_process
from ovqa.common.lavis.registry import registry
from ovqa.tasks.base_task import BaseTask
from ovqa.annotations.cls_templates import CLASSIFICATION_TEMPLATES
from packg.iotools import load_json, dump_json
from packg.paths import get_cache_dir


def get_classifier_cache_file(classnames, templates, model) -> Path:
    """
    Note: classifier will be the same if classnames, model class name, and model.loaded_ckpt_path
    are the same.

    """
    classnames_hash = hashlib.sha3_256(repr([classnames, templates]).encode()).hexdigest()
    model_info = [type(model).__name__, str(model)]

    # here we need to find some keys to get a unique model
    if hasattr(model, "loaded_ckpt_path"):
        model_info.append(model.loaded_ckpt_path)
    else:
        model_info.append(None)
    model_hash = hashlib.sha3_256(repr(model_info).encode()).hexdigest()
    cache_file = get_cache_dir() / f"classifier_cache_{classnames_hash}_{model_hash}.pth"
    return cache_file


def get_classnames_for_classifier(dataset, task_type):
    if task_type == MultimodalClassificationSynonymsTask and hasattr(dataset, "classsynonyms"):
        classsynonyms = dataset.classsynonyms
        classnames = list(itertools.chain.from_iterable(classsynonyms))
    else:
        classnames = dataset.classnames
    return classnames


def get_classnames_templates_for_classifier(dataset):
    if hasattr(dataset, "classtemplates"):
        classtemplates = dataset.classtemplates
    else:
        classtemplates = "openai_imagenet_template"
    assert (
        classtemplates in CLASSIFICATION_TEMPLATES.keys()
    ), f"{classtemplates} not available select one of the {list(CLASSIFICATION_TEMPLATES.keys())}"

    return CLASSIFICATION_TEMPLATES[classtemplates]


@registry.register_task("multimodal_classification")
class MultimodalClassificationTask(BaseTask):
    def __init__(self, save_embeddings_output_dir=None):
        super().__init__()
        self.save_embeddings_output_dir = save_embeddings_output_dir
        self.save_embeddings = self.save_embeddings_output_dir is not None

    @classmethod
    def setup_task(cls, **kwargs):
        cfg = kwargs["cfg"]
        save_embeddings_output_dir = None
        if cfg.run_cfg.get("save_embeddings", False):
            save_embeddings_output_dir = Path(cfg.run_cfg["output_dir"]) / "embeddings"
        return cls(save_embeddings_output_dir=save_embeddings_output_dir)

    def valid_step(self, model, samples):
        results = []

        if self.save_embeddings:
            outputs = model.predict(samples, return_embedding=True)
        else:
            outputs = model.predict(samples)
        embeddings = outputs["embeddings"].cpu() if self.save_embeddings else None

        logits = outputs["predictions"].cpu()
        targets = outputs["targets"]

        predictions = logits.max(1)[1].numpy()
        targets = targets.cpu().numpy()

        indices = samples[self.inst_id_key]

        for i, (pred, tgt, index, logi) in enumerate(zip(predictions, targets, indices, logits)):
            if isinstance(index, torch.Tensor):
                index = index.item()

            results.append(
                {
                    self.inst_id_key: index,
                    "prediction": pred.item(),
                    "target": tgt.item(),
                    "logits": logi,
                }
            )

            if self.save_embeddings:
                emb = embeddings.cpu()[i]
                file = Path(self.save_embeddings_output_dir) / f"{index}.pth"
                file.parent.mkdir(parents=True, exist_ok=True)
                print(f"Saving {index} to {file}")
                torch.save(emb, file)

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        # remove logits from results
        logit_dict = {v[self.inst_id_key]: v.pop("logits") for v in val_result}

        # save results as before, this also removes the duplicate keys
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate=self.inst_id_key,
        )

        # get the non-duplicate keys, save a logit array with the same order as those
        val_result_nodups = load_json(eval_result_file)
        inst_id_keys = [v[self.inst_id_key] for v in val_result_nodups]
        logits_list = [logit_dict[inst_id_key] for inst_id_key in inst_id_keys]
        logits_arr = torch.stack(logits_list, dim=0)
        logits_file = f"{Path(eval_result_file).as_posix()[:-5]}_logits.pth"
        torch.save(logits_arr, logits_file)

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, logits_file=logits_file, split_name=split_name
        )
        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, logits_file, split_name):
        results = json.load(open(eval_result_file))

        # compute acc1 and acc5 given logits
        targets = torch.tensor([res["target"] for res in results], dtype=torch.long)
        logits = torch.load(logits_file)
        num_classes = logits.shape[-1]
        acc1_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        acc5_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        softmaxed_logits = torch.softmax(logits.float(), dim=1)
        acc1 = acc1_metric(softmaxed_logits, targets).item()
        acc5 = acc5_metric(softmaxed_logits, targets).item()
        metrics = {"acc1": acc1, "acc5": acc5, "agg_metrics": acc1}

        log_stats = {split_name: {k: v for k, v in metrics.items()}}
        with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        logging.info(metrics)
        return metrics


@registry.register_task("multimodal_classification_synonyms")
class MultimodalClassificationSynonymsTask(MultimodalClassificationTask):
    def before_evaluation(self, model, dataset, **kwargs):
        # pass synonym list and templates to use

        model.before_evaluation(dataset=dataset, task_type=type(self))
        if hasattr(dataset, "classsynonyms"):
            classsynonyms = dataset.classsynonyms
            len_synonyms = [len(synonyms) for synonyms in classsynonyms]
        else:
            classnames = dataset.classnames
            len_synonyms = [1] * len(classnames)
            classsynonyms = [[classname] for classname in classnames]
        self.len_synonyms = len_synonyms

        # keep class synonyms and add them to the result folder
        self.classsynonyms_for_saving = classsynonyms

    def valid_step(self, model, samples):
        # save targets and logits here, compute predictions later
        results = []

        outputs = model.predict(samples)

        logits = outputs["predictions"].cpu()
        targets = outputs["targets"]

        indices = samples[self.inst_id_key]
        for tgt, index, logi in zip(targets, indices, logits):
            if isinstance(index, torch.Tensor):
                index = index.item()
            results.append(
                {
                    self.inst_id_key: index,
                    "target": tgt.item(),
                    "logits": logi,
                }
            )
        return results

    @main_process
    def _report_metrics(self, eval_result_file, logits_file, split_name):
        results = json.load(open(eval_result_file))
        logits = torch.load(logits_file)

        # split into synonyms
        logits_syn = logits.split(self.len_synonyms, dim=1)
        logits_maxsyn = []
        logits_avgsyn = []
        for x_syn in logits_syn:
            xavg_val = x_syn.mean(axis=1)
            logits_avgsyn.append(xavg_val)
            xmax_val, _ = x_syn.max(axis=1)
            logits_maxsyn.append(xmax_val)
        logits_avgsyn = torch.stack(logits_avgsyn, dim=1)
        logits_maxsyn = torch.stack(logits_maxsyn, dim=1)

        pred_avgsyn = logits_avgsyn.max(1)[1].numpy()
        pred_maxsyn = logits_maxsyn.max(1)[1].numpy()

        targets, indices = [], []
        for avgpred, maxpred, result in zip(pred_avgsyn, pred_maxsyn, results):
            result["prediction_max_syn"] = maxpred.item()
            result["prediction_avg_syn"] = avgpred.item()
            targets.append(result["target"])
            indices.append(result[self.inst_id_key])

        # # todo old calcuation, remove if the new one is ok
        # predictions_max_syn = np.array([res["prediction_max_syn"] for res in results])
        # predictions_avg_syn = np.array([res["prediction_avg_syn"] for res in results])
        # targets_np = np.array(targets)
        # accuracy_avg_syn = (targets_np == predictions_avg_syn).sum() / targets_np.shape[0]
        # accuracy_max_syn = (targets_np == predictions_max_syn).sum() / targets_np.shape[0]
        # metrics = {
        #     "agg_metrics": accuracy_max_syn,
        #     "acc_avg_syn": accuracy_avg_syn,
        #     "acc_max_syn": accuracy_max_syn,
        # }

        # new calculation including top5
        num_classes = logits_maxsyn.shape[-1]
        acc1_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        acc5_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        targets_torch = torch.tensor(targets, dtype=torch.long)
        metrics = {}
        for description, t_logits in zip(["max_syn", "avg_syn"], [logits_maxsyn, logits_avgsyn]):
            t_softmaxed_logits = torch.softmax(t_logits.float(), dim=1)
            acc1 = acc1_metric(t_softmaxed_logits, targets_torch).item()
            acc5 = acc5_metric(t_softmaxed_logits, targets_torch).item()
            metrics[f"acc1_{description}"] = acc1
            metrics[f"acc5_{description}"] = acc5
        metrics["agg_metrics"] = metrics["acc1_max_syn"]

        # update result json with predictions
        eval_result_file_base = Path(eval_result_file).as_posix()[:-5]
        eval_result_file_bkup = f"{eval_result_file_base}_raw.json"
        shutil.copy(eval_result_file, eval_result_file_bkup)
        dump_json(results, eval_result_file)

        syn_list_file = f"{eval_result_file_base}_synonym_list.json"
        dump_json(self.classsynonyms_for_saving, syn_list_file)

        log_stats = {split_name: {k: v for k, v in metrics.items()}}
        with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        logging.info(metrics)
        return metrics
