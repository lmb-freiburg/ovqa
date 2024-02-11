from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from transformers.data.metrics.squad_metrics import compute_f1

from ovqa.metrics.preprocessing import PrepC, get_preprocessing_fn
from ovqa.metrics.torchmetrics_ext import MetricExt


def get_metric_is_equal_default_prep():
    return TextComparison(
        comparison_fn=compare_is_equal,
        preproc_cand=PrepC.SIMPLE,
        preproc_ref=PrepC.SIMPLE,
    )


def get_metric_is_equal_vqa_prep():
    return TextComparison(
        comparison_fn=compare_is_equal,
        preproc_cand=PrepC.VQA_PRED,
        preproc_ref=PrepC.VQA_PRED,
    )


def get_metric_is_cont_default_prep():
    return TextComparison(
        comparison_fn=compare_is_contained,
        preproc_cand=PrepC.SIMPLE,
        preproc_ref=PrepC.SIMPLE,
    )


def get_metric_is_cont_vqa_prep():
    return TextComparison(
        comparison_fn=compare_is_contained,
        preproc_cand=PrepC.VQA_PRED,
        preproc_ref=PrepC.VQA_PRED,
    )


def get_f1_score_default_prep():
    return TextComparison(
        comparison_fn=compare_f1,
        preproc_cand=PrepC.SIMPLE,
        preproc_ref=PrepC.SIMPLE,
    )


def compare_is_equal(cand: str, ref: str):
    if cand == ref:
        return 1.0
    else:
        return 0.0


def compare_is_contained(cand: str, ref: str):
    if f" {ref} " in f" {cand} ":
        return 1.0
    else:
        return 0.0


def compare_f1(cand: str, ref: str):
    """Switch argument order to match all other comparison functions and ensure float."""
    return float(compute_f1(ref, cand))


def check_length_of_cand(cand: str, ref: str):
    return float(len(cand.split()))


class TextComparison(MetricExt):
    def __init__(
        self,
        comparison_fn=compare_is_equal,
        preproc_cand=PrepC.SIMPLE,
        preproc_ref=PrepC.SIMPLE,
        format_str="{:.2%}",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("cands", default=[], dist_reduce_fx="cat")
        self.add_state("refs", default=[], dist_reduce_fx="cat")
        self.comparison_fn = comparison_fn
        self.preproc_cand_fn = get_preprocessing_fn(preproc_ref)
        self.preproc_ref_fn = get_preprocessing_fn(preproc_cand)
        self.format_str = format_str

    def update(self, cands: list[str], refs: list[str], *args: list[str]) -> None:
        """Update state with data.

        Args:
            cands (list[str]): Predicted words.
            refs (list[str]): Targets words.
            *args (list[str]): Ignored arguments (e.g. question).
        """
        if len(cands) != len(refs):
            raise ValueError(
                f"Number of references and candidates must match but are "
                f"{len(refs)} and {len(cands)}"
            )
        self.cands.extend(cands)
        self.refs.extend(refs)

    def compute(self) -> float:
        """Compute the metric."""
        acc_list = self.compute_per_datapoint()
        acc = torch.mean(acc_list)
        return acc.item()

    def compute_per_datapoint(self, return_dict=False) -> torch.Tensor:
        """
        Returns:
            logits: shape (n_preds, n_classes)
        """
        acc_list = []
        for cand_raw, ref_raw in zip(self.cands, self.refs):
            cand = self.preproc_ref_fn(cand_raw)
            ref = self.preproc_cand_fn(ref_raw)
            acc = self.comparison_fn(cand, ref)
            acc_list.append(acc)
        scores = torch.tensor(acc_list, device=self.device)
        if return_dict:
            return {"scores": scores}
        return scores

    def format(self, value: float):
        return self.format_str.format(value)

    def close(self):
        pass


class TextComparisonSynonyms(TextComparison):
    def __init__(
        self,
        comparison_fn=compare_is_equal,
        preproc_cand=PrepC.SIMPLE,
        preproc_ref=PrepC.SIMPLE,
        name2syn: Optional[dict[str, list[str]]] = None,
        syn_mode: str = "arg_max_syn",
        min_acc_for_word_list: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__(
            comparison_fn=comparison_fn,
            preproc_cand=preproc_cand,
            preproc_ref=preproc_ref,
            **kwargs,
        )
        if name2syn is None:
            name2syn = {}
        self.name2syn = name2syn
        self.syn_mode = syn_mode
        self.min_acc_for_word_list = min_acc_for_word_list

    def compute(self) -> float:
        acc_list, _ = self.compute_per_datapoint()
        acc = torch.mean(acc_list)
        return acc.item()

    def compute_per_datapoint(self, return_dict=False) -> tuple[torch.Tensor, list[str]]:
        acc_list, contain_word_list = [], []
        for cand_raw, ref_raw in zip(self.cands, self.refs):
            cand = self.preproc_ref_fn(cand_raw)
            ref = self.preproc_cand_fn(ref_raw)
            ref_syns_raw = self.name2syn.get(ref_raw, [])
            ref_syns = [self.preproc_ref_fn(text) for text in ref_syns_raw] + [ref]

            accs = []
            for ref_text in ref_syns:
                accs.append(self.comparison_fn(cand, ref_text))
            max_acc_idx = np.argmax(accs)
            max_acc = accs[max_acc_idx]
            max_word = ref_syns[max_acc_idx]
            if max_acc <= self.min_acc_for_word_list:
                # if the max accuracy is too small, instead return empty string as max_word
                # e.g. if contain metric matched no word and acc is 0.0
                max_word = ""

            if self.syn_mode == "arg_max_syn":
                acc = max_acc
            elif self.syn_mode == "average_syn":
                acc = np.mean(accs)
            else:
                raise ValueError(f"syn_mode {self.syn_mode} not supported")

            acc_list.append(acc)
            contain_word_list.append(max_word)

        scores = torch.tensor(acc_list, device=self.device)
        if return_dict:
            return {
                "scores": scores,
                "contain_word_list": contain_word_list,
            }
        return scores, contain_word_list
