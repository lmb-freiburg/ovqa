from __future__ import annotations

import torch

from ovqa.metrics.preprocessing import PrepC, get_preprocessing_fn
from ovqa.metrics.vqa.base_vqa_metric import BaseVqaMetric


class VqaPredLen(BaseVqaMetric):
    """
    Measure length of prediction in words.
    """

    def __init__(
        self,
        preproc_pred: str | list[str] = PrepC.VQA_PRED,
        preproc_target: str | list[str] = PrepC.VQA_ANSWER,
    ):
        super().__init__()
        self.preproc_pred_fn = get_preprocessing_fn(preproc_pred)
        self.preproc_target_fn = get_preprocessing_fn(preproc_target)

    def compute_per_answer(self) -> torch.Tensor:
        accs_tensor = self.compute_per_datapoint()
        return [[a] for a in accs_tensor.tolist()]

    def aggregate_vqa_scores(self, scores_per_answer):
        return [a[0] for a in scores_per_answer]

    def compute_per_datapoint(self, return_dict=False) -> torch.Tensor:
        preds_proc = [self.preproc_pred_fn(pred) for pred in self.pred_values]
        accs = []
        for pred in preds_proc:
            accs.append(float(len(pred.split())))
        acc_tensor = torch.tensor(accs)
        if return_dict:
            return {"scores": acc_tensor}
        return acc_tensor

    def format(self, value: float):
        return f"{value:.2f}"
