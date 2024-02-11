import json
import logging
import numpy as np
import os

import ovqa.common.lavis.dist_utils as dist_utils
from ovqa.datasets.classifier_vqa_dataset import ClassifierVQADataset
from ovqa.outputs import QAOutput
from ovqa.tasks.vqa_task_utils import (
    save_vqa_output,
    after_predict_answers_valid_step,
)
from ovqa.common.lavis.registry import registry
from ovqa.tasks.base_task import BaseTask
from ovqa.tasks.vqa import get_generation_kwargs
from packg.iotools.jsonext import load_json


@registry.register_task("multilabel_classifier_vqa")
class MultilabelClassifierVQATask(BaseTask):
    def __init__(self, run_cfg):
        super().__init__()
        self.config = run_cfg
        self.evaluate = run_cfg.get("evaluate", False)
        self.generation_kwargs = get_generation_kwargs(run_cfg)
        self.answer_list = None
        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        return cls(run_cfg)

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)
        self.annotation, self.answer_list = get_anno_for_classifier_vqa(datasets)
        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."
        return datasets

    def valid_step(self, model, samples):
        qa_output: QAOutput = model.predict_multiquestion_answers(
            samples=samples,
            return_dict=True,
            answer_list=self.answer_list,
            **self.generation_kwargs,
        )
        return after_predict_answers_valid_step(samples, qa_output)

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = save_vqa_output(self, val_result, split_name, vqa_field_name="answers")
        # metrics = self._report_metrics(result_file=result_file, split=split_name)
        metrics = []
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Calculate accuracy
        """
        anno = self.annotation[split]
        anno_dict = convert_list_to_dict(anno, "class_idx")

        results = load_json(result_file)  # list of {"question_id": int, "answer": str}
        results_dict = convert_list_to_dict(results, "answers")

        labels = self.answer_list
        metrics = eval_classifier_vqa(results_dict, anno_dict, labels)

        with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
            f.write(json.dumps(metrics) + "\n")

        return metrics


def get_anno_for_classifier_vqa(datasets):
    annotation = {}
    answer_list = None
    for dataset in datasets.values():
        for split in dataset:
            dset: ClassifierVQADataset = dataset[split]
            annotation[split] = dset.annotation
            answer_list = dataset[split].answer_list
    assert answer_list is not None, "Answer list is not available."
    return annotation, answer_list


def convert_list_to_dict(list_data, value_field, key_field="question_id"):
    dict_data = {str(item[key_field]): item[value_field] for item in list_data}
    return dict_data


def eval_classifier_vqa(results_dict, anno_dict, labels):
    """

    Args:
        results_dict: dict {question_id: answer}
        anno_dict: dict {question_id: class_idx}
        labels: list of label str

    Returns:
        metrics dict
    """

    # for the answer score style eval the labels are the answer list
    # and each answer is one of the label

    metrics = {}
    label2idx = {label: idx for idx, label in enumerate(labels)}

    # naive eval: either hit the label perfectly or the answer is wrong.
    pred_labels = np.array([label2idx.get(answer, -1) for qid, answer in results_dict.items()])

    # HACK to correct int and str missmatch type
    # print("anno_dict", list(anno_dict.keys())[0], type(list(anno_dict.keys())[0]))
    # print("results_dict", list(results_dict.keys())[0], type(list(results_dict.keys())[0]))
    if type(list(anno_dict.keys())[0]) != type(list(results_dict)[0]):
        # convert everything to str
        if isinstance(list(anno_dict.keys())[0], int):
            new_anno_dict = {str(key): val for key, val in anno_dict.items()}
            anno_dict = new_anno_dict
            del new_anno_dict
        if isinstance(list(results_dict.keys())[0], int):
            new_results_dict = {str(key): val for key, val in results_dict.items()}
            results_dict = new_results_dict
            del new_results_dict

    assert list(anno_dict.keys()) == list(results_dict.keys()), (
        f"Mismatched keys between predictions and annotations."
        f"\n========== annotations:\n{anno_dict}\n\n========== predictions:\n{results_dict}"
    )
    gt_labels = np.array([label for qid, label in anno_dict.items()])
    acc = np.mean(gt_labels == pred_labels)
    metrics["acc"] = acc

    for i, (k, class_idx) in enumerate(anno_dict.items()):
        if i >= 5:
            break
        result = results_dict[k]
        label = labels[class_idx]
        logging.info(f"Example {i} {k}: pred {result} gt {label}")

    n_invalid = np.sum(pred_labels == -1)
    n_total = len(pred_labels)

    logging.info(f"Invalid answer count: {n_invalid}/{n_total} ({n_invalid / n_total:.2%})")
    logging.info("Classification accuracy is: %.02f\n" % acc)
    return metrics
