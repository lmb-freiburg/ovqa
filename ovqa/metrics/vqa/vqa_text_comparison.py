from __future__ import annotations

import torch

from ovqa.metrics.preprocessing import PrepC, get_preprocessing_fn
from ovqa.metrics.simple import compare_is_equal
from ovqa.metrics.torchmetrics_ext import MetricExt
from ovqa.metrics.vqa.base_vqa_metric import aggregate_vqa_scores


class VqaTextComparison(MetricExt):
    def __init__(
        self,
        comparison_fn=compare_is_equal,
        hit_value=0.3,
        preproc_pred: str | list[str] = PrepC.VQA_PRED,
        preproc_target: str | list[str] = PrepC.VQA_PRED,
        format_str="{:.2%}",
    ):
        super().__init__()
        self.add_state("pred_values", default=[], dist_reduce_fx="cat")
        self.add_state("answers", default=[], dist_reduce_fx="cat")
        self.hit_value = hit_value
        self.preproc_pred_fn = get_preprocessing_fn(preproc_pred)
        self.preproc_target_fn = get_preprocessing_fn(preproc_target)
        self.format_str = format_str
        self.comparison_fn = comparison_fn

    def update(self, values: list[str], answers: list[list[str]], *args) -> None:
        """Update state with data.

        Args:
            values: Prediction values (model output).
            answers: Ground truth answers.
            *args: Ignored arguments (e.g. questions)
        """
        self.pred_values.extend(values)
        self.answers.extend(answers)

    def compute(self) -> float:
        raise NotImplementedError

    def compute_per_answer(self) -> list[list[float]]:
        preds_proc = [self.preproc_pred_fn(pred) for pred in self.pred_values]
        answer_lists_proc = [[self.preproc_target_fn(a) for a in alist] for alist in self.answers]
        accs = []
        for pred, answer_list in zip(preds_proc, answer_lists_proc):
            accs_here = []
            for answer in answer_list:
                accs_here.append(self.comparison_fn(pred, answer))
            accs.append(accs_here)
        return accs

    def compute_per_datapoint(self) -> torch.Tensor:
        """
        Returns:
            accuracy per datapoint: shape (n_preds,)
        """
        scores_per_answer = self.compute_per_answer()
        scores_per_datapoint = self.aggregate_vqa_scores(scores_per_answer)
        return torch.tensor(scores_per_datapoint, device=self.device)

    def aggregate_vqa_scores(self, scores_per_answer):
        """"Turn per-answer scores into per-datapoint scores"""
        return aggregate_vqa_scores(scores_per_answer, hit_value=self.hit_value)

    def format(self, value: float):
        return self.format_str.format(value)

    def close(self):
        pass
