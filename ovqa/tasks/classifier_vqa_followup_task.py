from collections import Counter

import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import ovqa.common.lavis.dist_utils as dist_utils
from ovqa.common.lavis.registry import registry
from ovqa.datasets.imagenet_hierarchy import load_hierarchy

from ovqa.followup import Followup
from ovqa.datasets.classifier_vqa_dataset import ClassifierVQADataset
from ovqa.tasks.classifier_vqa_task import (
    ClassifierVQATask,
    convert_list_to_dict,
    eval_classifier_vqa,
)
from ovqa.tasks.vqa_task_utils import save_vqa_output
from ovqa.result_loader import read_single_result
from packg.iotools.jsonext import load_json, dump_json


@registry.register_task("classifier_vqa_followup")
class ClassifierVQAFollowupTask(ClassifierVQATask):
    def __init__(self, cfg):
        cfg.run_cfg["prompt"] = cfg.run_cfg["followup_cfg"]["followup_prompt"]
        logging.info(f"Modified run cfg prompt: {cfg.run_cfg['prompt']}")
        super().__init__(cfg)
        self.default_followup_object = cfg.run_cfg["followup_cfg"].get(
            "default_followup_object", "object"
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)
        run_cfg = cfg.run_cfg

        # load followup config
        followup_cfg = run_cfg["followup_cfg"]
        logging.info(f"followup_cfg: {followup_cfg}")

        # find full name of current dataset
        test_splits = run_cfg["test_splits"]
        assert len(datasets) == 1, f"Only one dataset supported, got {datasets}"
        dataset_name = list(datasets.keys())[0]
        assert len(test_splits) == 1, f"Only one test split supported, got {test_splits}"
        dataset_split = test_splits[0]

        # # here we get "imagenet1k", maybe we will need "imagenet1k-square" later:
        # dataset_cfg = cfg.datasets_cfg[dataset_name]
        # dataset_name_full = dataset_name
        # cropped_image_dir = dataset_cfg.get("cropped_image_dir", None)
        # if cropped_image_dir is not None:
        #     dataset_name_full = f"{dataset_name}-{cropped_image_dir}"

        # load classsynonyms from dataset if required in the config
        dataset: ClassifierVQADataset = datasets[dataset_name][dataset_split]
        classsynonyms = dataset.classsynonyms
        if classsynonyms is None:
            synonym_dict = None
            assert (
                not followup_cfg.use_synonyms_leaves
            ), "classsynonyms not found, but use_synonyms_leaves is true"
        else:
            synonym_dict = {name: i for i, names in enumerate(classsynonyms) for name in names}

        # load hierarchy
        hier = load_hierarchy(dataset_name)

        # create followup class
        targets = {v["key"]: v["class_idx"] for v in dataset.annotation}
        follower = Followup(followup_cfg, hier, dataset.classnames, synonym_dict, targets)

        # load existing predictions
        followup_prev_dir = run_cfg["followup_prev_dir"]
        result_obj = read_single_result(followup_prev_dir)
        assert result_obj is not None, f"Failed to read output from: {followup_prev_dir}"
        preds = result_obj.load_output()

        self.datapoint_num2key = {}
        if next(iter(targets.keys())) not in preds:
            # fix prediction keys from '0' to ''val_00000001' etc
            new_preds = {}
            for i, v in enumerate(dataset.annotation):
                key = v["key"]
                pred = preds[str(i)]
                new_preds[key] = pred
                self.datapoint_num2key[i] = key
            preds = new_preds

        self.old_preds = preds
        to_followup = follower.evaluate_pipeline(preds)
        # to_followup now looks like
        # {'val_00000003': {'status': 'followup', 'object': 'dog'},} ...
        # where status is "correct", "failed" or "followup" and in case of followup "object" is set.
        counter_followup = Counter(v["status"] for v in to_followup.values())
        logging.info(str(dict(counter_followup)))
        self.tosave_followup_counter = counter_followup

        # output_dir is not registered until later so we cannot save here, save for later
        self.tosave_followup_results = to_followup

        # update dataset and config based on this
        new_anns = []
        for ann in dataset.annotation:
            ann_followup = to_followup[ann["key"]]
            if ann_followup["status"] in "correct":
                continue
            # define the followup question
            if ann_followup["status"] == "followup":
                ask_object = ann_followup[self.default_followup_object]
            elif ann_followup["status"] == "failed":
                ask_object = self.default_followup_object
            else:
                raise ValueError(f"Unknown status: {ann_followup['status']}")
            new_ann = deepcopy(ann)
            # note this is used in ClassifierVQADataset.get_item
            new_ann["question_followup"] = ask_object

            new_anns.append(new_ann)

        dataset.annotation = new_anns
        logging.info(f"Updated dataset, new length: {len(dataset.annotation)}")

        return datasets

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = save_vqa_output(self, val_result, split_name, file_identifier="followup")
        metrics = self._report_metrics(result_file=result_file, split=split_name)
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        # save results of followup analysis (status of old answers and the followup questions)
        dump_json(
            self.tosave_followup_results,
            Path(registry.get_path("output_dir")) / "result/followup.json",
            verbose=False,
        )
        # save old predictions
        dump_json(
            self.old_preds,
            Path(registry.get_path("output_dir")) / "result/old_preds.json",
            verbose=False,
        )
        # save counter
        dump_json(
            self.tosave_followup_counter,
            Path(registry.get_path("output_dir")) / "result/followup_count.json",
            verbose=False,
        )

        anno = self.annotation[split]
        anno_dict = convert_list_to_dict(anno, "class_idx")

        followup_results = load_json(result_file)  # list of {"question_id": int, "answer": str}
        followup_results_dict = convert_list_to_dict(followup_results, "answer")

        old_preds = self.old_preds

        # aggregate old and new answers
        if next(iter(followup_results_dict.keys())) not in old_preds:
            # fix keys again
            if len(self.datapoint_num2key) == 0:
                t1 = list(old_preds.keys())
                t2 = list(followup_results_dict.keys())
                raise RuntimeError(
                    f"self.num2key not set but keys incorrect: "
                    f"old preds type ({type(t1[0])}) values {t1[:10]}... "
                    f"new results type ({type(t2[0])}) values {t2[:10]}... "
                )
            # we want to save in lavis format so now we have to translate
            # from "val_00000001" back to "0"
            key2num = {v: int(k) for k, v in self.datapoint_num2key.items()}
            old_preds = {str(key2num[k]): v for k, v in old_preds.items()}

        final_results_dict = {}
        c_new = 0
        for numstr, old_answer in old_preds.items():
            if numstr in followup_results_dict:
                final_results_dict[numstr] = followup_results_dict[numstr]
                c_new += 1
                continue
            final_results_dict[numstr] = old_answer
        logging.info(f"Update {c_new} new answers, total {len(final_results_dict)} answers")

        # evaluate the new answers only
        labels = self.answer_list
        metrics = eval_classifier_vqa(final_results_dict, anno_dict, labels)
        with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # save aggregated output in lavis format
        final_results = []
        for qid in sorted(int(a) for a in final_results_dict.keys()):
            final_results.append(
                {
                    "answer": final_results_dict[str(qid)],
                    "question_id": qid,
                }
            )

        dump_json(
            final_results,
            Path(registry.get_path("output_dir")) / f"result/{split}_vqa_result.json",
            verbose=False,
        )

        return metrics
