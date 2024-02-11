"""
Implementation of various ngram based metrics.

Code adapted from https://github.com/salaniz/pycocoevalcap
Original code license: https://github.com/salaniz/pycocoevalcap/blob/master/license.txt
"""

from __future__ import annotations

import os
import sys
import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from torchmetrics import Metric

from packg import Const
from ovqa.metrics.preprocessing import PrepC, get_preprocessing_fn


class CocoScorerC(Const):
    METEOR = "METEOR"
    ROUGE = "ROUGE"
    CIDER = "CIDEr"
    SPICE = "SPICE"
    BLEU1 = "Bleu_1"
    BLEU2 = "Bleu_2"
    BLEU3 = "Bleu_3"
    BLEU4 = "Bleu_4"


nonbleu_scorer_map = {
    CocoScorerC.METEOR: Meteor,
    CocoScorerC.ROUGE: Rouge,
    CocoScorerC.CIDER: Cider,
    CocoScorerC.SPICE: Spice,
}


class NgramMetric(Metric):
    def __init__(
        self,
        scorer_name: str,
        preproc_cand=PrepC.NONE,
        preproc_ref=PrepC.NONE,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("cands", default=[], dist_reduce_fx="cat")
        self.add_state("refs", default=[], dist_reduce_fx="cat")
        self.preproc_cand_fn = get_preprocessing_fn(preproc_ref)
        self.preproc_ref_fn = get_preprocessing_fn(preproc_cand)
        self.scorer_name = scorer_name

    def update(self, cands: list[str], refs: list[str]) -> None:
        """Update state with data.

        Args:
            cands (list[str]): Predicted words.
            refs (list[str]): Targets words.
        """
        self.cands.extend(cands)
        self.refs.extend(refs)

    def compute_aggregated_and_per_datapoint(self) -> tuple[float, list[float]]:
        """Compute the metric."""
        assert len(self.refs) == len(self.cands), (
            f"Number of references and candidates must match but are "
            f"{len(self.refs)} and {len(self.cands)}"
        )
        refs_dict = {i: [ref] for i, ref in enumerate(self.refs)}
        cands_dict = {i: [cand] for i, cand in enumerate(self.cands)}

        if self.scorer_name.lower().startswith("bleu"):
            n = int(self.scorer_name[-1])
            scorer = Bleu(n)
            score_mult, scores_mult = scorer.compute_score(refs_dict, cands_dict, verbose=1)
            score = score_mult[-1]
            scores = scores_mult[-1]
            return score, scores

        scorer = nonbleu_scorer_map[self.scorer_name]()

        if self.scorer_name == CocoScorerC.SPICE:
            score, scores = compute_score_silent(scorer, refs_dict, cands_dict)
            # spice outputs a bunch of things for each datapoint
            # final score is average of "All" "f" so this is the relevant one
            scores = [s["All"]["f"] for s in scores]
            return score, scores

        score, scores = scorer.compute_score(refs_dict, cands_dict)
        return score, scores

    def compute(self) -> float:
        score, _ = self.compute_aggregated_and_per_datapoint()
        return score

    def compute_per_datapoint(self, return_dict=False) -> torch.Tensor:
        _, scores = self.compute_aggregated_and_per_datapoint()
        scores = torch.tensor(scores)
        if return_dict:
            return {"scores": scores}
        return scores

    def format(self, value: float):
        return f"{value:.3f}"

    def close(self):
        pass


def compute_score_silent(scorer, refs_dict, cands_dict):
    # suppress spice warnings
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)
    with open(os.devnull, "wb") as null_file:
        null_fd = null_file.fileno()
        os.dup2(null_fd, original_stdout_fd)
        os.dup2(null_fd, original_stderr_fd)
        try:
            score, scores = scorer.compute_score(refs_dict, cands_dict)
        finally:
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
    return score, scores


def main():
    refs = ["The cat is on the mat.", "There is a cat on the mat."]
    cands = refs + ["The The The The The.", "Mat the cat is on a there."]

    refs_list, cands_list = [], []
    for r in refs:
        for c in cands:
            refs_list.append(r)
            cands_list.append(c)
    for scorer_name in CocoScorerC.values():
        print(f"========== {scorer_name} ==========")
        metric = NgramMetric(scorer_name)
        metric.update(cands_list, refs_list)
        score, scores = metric.compute_aggregated_and_per_datapoint()
        print(f"{scorer_name}: {score:.3f}")
        for r, c, s in zip(refs_list, cands_list, scores):
            print(f"  {s:.3f} for {r} ===VS=== {c}")


if __name__ == "__main__":
    main()
