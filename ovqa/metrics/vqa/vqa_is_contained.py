from __future__ import annotations

import torch

from ovqa.metrics.preprocessing import PrepC, get_preprocessing_fn
from ovqa.metrics.vqa.base_vqa_metric import BaseVqaMetric, aggregate_vqa_scores


class VqaIsContainedAccuracy(BaseVqaMetric):
    """
    todo this could be simplified by just having a comparison function that is EM or cont,
        and then we can just pass in the function as a parameter
        instead of having two metrics.

    Difference to normal contained: Here we have N GT answers instead of one
    and calculate a weighted soft accuracy over them.

    Formula: accuracy = min(1, num_matches * hit_value)

    Attributes:
        hit_value: How much accuracy is gained when matching one of the human answers.
            Depends on the dataset:
                vqav2: 10 answers, hit_value=0.3
                ivqa: 5 answers, hit_value=0.5
    """

    def __init__(
        self,
        hit_value=0.3,
        preproc_pred: str | list[str] = PrepC.SIMPLE,
        preproc_target: str | list[str] = PrepC.SIMPLE,
    ):
        super().__init__()
        self.hit_value = hit_value
        self.preproc_pred_fn = get_preprocessing_fn(preproc_pred)
        self.preproc_target_fn = get_preprocessing_fn(preproc_target)

    def compute_per_answer(self) -> list[list[float]]:
        preds_proc = [self.preproc_pred_fn(pred) for pred in self.pred_values]
        answer_lists_proc = [[self.preproc_target_fn(a) for a in alist] for alist in self.answers]
        accs = []
        for pred, answer_list in zip(preds_proc, answer_lists_proc):
            accs_here = []
            for answer in answer_list:
                if f" {answer} " in f" {pred} ":
                    accs_here.append(1.0)
                else:
                    accs_here.append(0.0)
            accs.append(accs_here)
        return accs

    def compute_per_datapoint(self) -> torch.Tensor:
        """
        Returns:
            accuracy per datapoint: shape (n_preds,)
        """
        scores_per_answer = self.compute_per_answer()
        scores_per_datapoint = aggregate_vqa_scores(scores_per_answer, self.hit_value)
        return torch.tensor(scores_per_datapoint, device=self.device)
