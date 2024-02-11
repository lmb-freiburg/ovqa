from __future__ import annotations

import torch
from typing import Optional

from ovqa.metrics.preprocessing import PrepC, get_preprocessing_fn
from ovqa.metrics.vqa.base_vqa_metric import BaseVqaMetric, aggregate_vqa_scores
from ovqa.textutils.embeddings import EmbeddingsPackageConst, get_sentence_embedder


class VqaMatchEmbedding(BaseVqaMetric):
    """
    1. Binary match each answer

    States for reference
        self.add_state("pred_values", default=[], dist_reduce_fx="cat")
        self.add_state("pred_keys", default=[], dist_reduce_fx="cat")
        self.add_state("answers", default=[], dist_reduce_fx="cat")
    """

    def __init__(
        self,
        package_name=EmbeddingsPackageConst.OPEN_CLIP,
        embedder_name="EVA01-g-14/laion400m_s11b_b41k",
        templates_name="none",
        model_kwargs=None,
        average_over_sentences: Optional[str] = None,
        hit_value=0.3,
        preproc_pred: str | list[str] = PrepC.SIMPLE,
        preproc_target: str | list[str] = PrepC.SIMPLE,
    ):
        super().__init__()
        self.hit_value = hit_value
        self.preproc_pred_fn = get_preprocessing_fn(preproc_pred)
        self.preproc_target_fn = get_preprocessing_fn(preproc_target)

        self.package_name = package_name
        self.embedder_name = embedder_name
        self.templates_name = templates_name
        self.average_over_sentences = average_over_sentences

        model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.embedder = get_sentence_embedder(
            model_name=embedder_name, package_name=package_name, **model_kwargs
        )

    def compute_per_datapoint(self) -> torch.Tensor:
        """
        Returns:
            accuracy per datapoint: shape (n_preds,)
        """
        answer_lists_proc = [[self.preproc_target_fn(a) for a in alist] for alist in self.answers]
        logits = self.compute_logits()
        top1_ids = torch.argmax(logits, dim=1)
        preds_proc = [self.class_names[i] for i in top1_ids]
        accs = []
        for pred, answer_list in zip(preds_proc, answer_lists_proc):
            accs_here = []
            for answer in answer_list:
                if answer == pred:
                    accs_here.append(1.0)
                else:
                    accs_here.append(0.0)
            accs.append(accs_here)

        acc_list = aggregate_vqa_scores(accs, hit_value=self.hit_value)
        return torch.tensor(acc_list, device=self.device)
