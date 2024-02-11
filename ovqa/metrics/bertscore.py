import torch
from torchmetrics import Metric

from ovqa.metrics.preprocessing import PrepC, get_preprocessing_fn


class BertscoreMetric(Metric):
    def __init__(
        self,
        model_name=None,
        lang="en",
        preproc_cand=PrepC.NONE,
        preproc_ref=PrepC.NONE,
        rescale_with_baseline: bool = True,
        device="cuda",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("cands", default=[], dist_reduce_fx="cat")
        self.add_state("refs", default=[], dist_reduce_fx="cat")
        self.preproc_cand_fn = get_preprocessing_fn(preproc_ref)
        self.preproc_ref_fn = get_preprocessing_fn(preproc_cand)
        self.model_name = model_name
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline
        self._model_device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import bert_score

            self._model = bert_score.BERTScorer(
                model_type=self.model_name,
                lang=self.lang,
                rescale_with_baseline=self.rescale_with_baseline,
                device=self._model_device,
            )
        return self._model

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
        p, r, f = self.model.score(cands_input, refs_input)
        if return_dict:
            return {"scores": f}
        return f  # torch.tensor(f, device=self.device)

    def close(self):
        self.scorer = None


def main():
    # simple example
    cands = ["hello there", "general kenobi"]
    refs = ["hello there", "general hunzone"]
    metric = BertscoreMetric(model_name="bert-base-uncased")
    metric.update(cands, refs)
    print(metric.compute_per_datapoint())


if __name__ == "__main__":
    main()
