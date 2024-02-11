"""
Usage:

    >>> # latest and biggest version
    >>> bleurt1 = BleurtMetric()
    >>> # latest but smaller and faster
    >>> bleurt2 = BleurtMetric(model_name="lucadiliello/BLEURT-20")
    >>> # old original bleurt
    >>> bleurt3 = BleurtMetric(model_name="Elron/bleurt-large-512")

From the original BLEURT paper:
    We experiment with two versions of BLEURT, BLEURT and BLEURTbase,
    respectively based on BERT- Large (24 layers, 1024 hidden units, 16 heads)
    and BERT-Base (12 layers, 768 hidden units, 12 heads)

Normalization of values:
    BLEURT (v1) outputs values around -2 to 1
    BLEURT-20 (v2) outputs values around 0 to 1
"""

import torch
from ovqa.metrics.bleurt_pytorch import (
    BleurtConfig,
    BleurtTokenizer,
    BleurtForSequenceClassification,
)
from math import ceil
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ovqa.metrics.preprocessing import PrepC, get_preprocessing_fn
from ovqa.metrics.torchmetrics_ext import MetricExt


class BleurtMetric(MetricExt):
    def __init__(
        self,
        model_name="lucadiliello/BLEURT-20",
        preproc_cand=PrepC.NONE,
        preproc_ref=PrepC.NONE,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("cands", default=[], dist_reduce_fx="cat")
        self.add_state("refs", default=[], dist_reduce_fx="cat")
        self.preproc_cand_fn = get_preprocessing_fn(preproc_ref)
        self.preproc_ref_fn = get_preprocessing_fn(preproc_cand)
        self.model_name = model_name
        wrapper_class = MODEL2CLASS[self.model_name]
        self.bleurt = wrapper_class(
            model_name=self.model_name, device=self.device, normalize=normalize
        )

    def update(self, cands: list[str], refs: list[str]) -> None:
        self.cands.extend(cands)
        self.refs.extend(refs)

    def compute(self) -> float:
        """Compute the metric."""
        score_list = self.compute_per_datapoint()
        score = torch.mean(score_list)
        return score.item()

    def compute_per_datapoint(self, return_dict=False) -> torch.Tensor:
        """
        Returns:
            logits: shape (n_preds, n_classes)
        """
        cands_input, refs_input = [], []
        for cand_raw, ref_raw in zip(self.cands, self.refs):
            cand = self.preproc_ref_fn(cand_raw)
            ref = self.preproc_cand_fn(ref_raw)
            cands_input.append(cand)
            refs_input.append(ref)
        score = self.bleurt.score(refs_input, cands_input)
        if isinstance(score, torch.Tensor):
            score = score.clone().detach().to(self.device)
        else:
            score = torch.tensor(score, device=self.device)

        if return_dict:
            return {"scores": score}
        return score

    def close(self):
        self.bleurt.close()


class BleurtWrapper:
    def __init__(
        self,
        model_name="Elron/bleurt-large-512",
        device: str = "cuda",
        batch_size: int = 16,
        normalize: bool = True,
        norm_min=-2,
        norm_max=1,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = None
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.norm_min = norm_min
        self.norm_max = norm_max

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)
        return self._model

    def score(self, references: list[str], candidates: list[str]):
        assert len(references) == len(candidates), (
            f"Number of references and candidates must match. Got {len(references)} references "
            f"and {len(candidates)} candidates."
        )
        n_total = len(references)
        n_batches = ceil(n_total / self.batch_size)
        all_batches = []
        with torch.no_grad():
            for n_batch in range(n_batches):
                start = n_batch * self.batch_size
                end = min((n_batch + 1) * self.batch_size, n_total)
                batch_references = references[start:end]
                batch_candidates = candidates[start:end]
                tokenizer_output = self.tokenizer(
                    batch_references, batch_candidates, return_tensors="pt", padding=True
                )
                scores = self.model(**tokenizer_output.to(self.model.device))[0].squeeze()
                if scores.ndim == 0:
                    scores = scores.unsqueeze(0)
                all_batches.append(scores.cpu())
        scores = torch.cat(all_batches)
        if len(scores) != len(references):
            raise ValueError(
                f"Number of scores does not match number of references. Got "
                f"{scores.shape} shaped scores {scores} and {len(references)} references."
                f"Candidates: {candidates} References: {references}"
            )

        if self.normalize:
            scores = (scores - self.norm_min) / (self.norm_max - self.norm_min)
            scores = scores.clamp(0, 1)
        return scores

    def close(self):
        self._model = None


class Bleurt20Wrapper:
    def __init__(
        self,
        model_name="lucadiliello/BLEURT-20",
        device: str = "cuda",
        batch_size: int = 16,
        normalize: bool = False,
        norm_min=0,
        norm_max=1,
    ):
        self._model = None
        self.device = device
        self.config = BleurtConfig.from_pretrained(model_name)
        self.tokenizer = BleurtTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.norm_min = norm_min
        self.norm_max = norm_max

    @property
    def model(self):
        if self._model is None:
            self._model = BleurtForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)
        return self._model

    def score(self, references: list[str], candidates: list[str]):
        assert len(references) == len(candidates), (
            f"Number of references and candidates must match. Got {len(references)} references "
            f"and {len(candidates)} candidates."
        )
        n_total = len(references)
        n_batches = ceil(n_total / self.batch_size)
        all_batches = []
        with torch.no_grad():
            for n_batch in range(n_batches):
                start = n_batch * self.batch_size
                end = min((n_batch + 1) * self.batch_size, n_total)
                batch_references = references[start:end]
                batch_candidates = candidates[start:end]
                tokenizer_output = self.tokenizer(
                    batch_references, batch_candidates, padding="longest", return_tensors="pt"
                )
                scores = self.model(**tokenizer_output.to(self.model.device)).logits.flatten()
                if scores.ndim == 0:
                    scores = scores.unsqueeze(0)
                all_batches.append(scores.cpu())

        scores = torch.cat(all_batches)
        if len(scores) != len(references):
            raise ValueError(
                f"Number of scores does not match number of references. Got "
                f"{scores.shape} shaped scores {scores} and {len(references)} references."
                f"Candidates: {candidates} References: {references}"
            )
        if self.normalize:
            scores = (scores - self.norm_min) / (self.norm_max - self.norm_min)
            scores = scores.clamp(0, 1)
        return scores

    def close(self):
        self._model = None


MODEL2CLASS = {
    "Elron/bleurt-base-128": BleurtWrapper,
    "Elron/bleurt-base-512": BleurtWrapper,
    "Elron/bleurt-large-128": BleurtWrapper,
    "Elron/bleurt-large-512": BleurtWrapper,
    "lucadiliello/BLEURT-20": Bleurt20Wrapper,
    "lucadiliello/BLEURT-20-D12": Bleurt20Wrapper,
}


def main():
    references = ["hello world", "hello world"]
    candidates = ["hi universe", "bye world"]
    bleurt = BleurtWrapper()
    scores = bleurt.score(references, candidates)
    print(scores)  # tensor([0.9877, 0.0475])


if __name__ == "__main__":
    main()
