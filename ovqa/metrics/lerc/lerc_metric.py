import torch
from torchmetrics import Metric
from typing import Optional

from visiontext.cacheutils import CachePlugin
from ovqa.metrics.lerc.lerc_model.lerc_predictor import get_pretrained_lerc
from ovqa.metrics.preprocessing import PrepC, get_preprocessing_fn


class LercMetric(Metric):
    def __init__(
        self,
        preproc_cand=PrepC.NONE,
        preproc_ref=PrepC.NONE,
        preproc_question=PrepC.NONE,
        preproc_context=PrepC.NONE,
        device="cuda",
        use_cache: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model = None
        self.add_state("cands", default=[], dist_reduce_fx="cat")
        self.add_state("refs", default=[], dist_reduce_fx="cat")
        self.add_state("questions", default=[], dist_reduce_fx="cat")
        self.add_state("contexts", default=[], dist_reduce_fx="cat")
        self.preproc_cand_fn = get_preprocessing_fn(preproc_ref)
        self.preproc_ref_fn = get_preprocessing_fn(preproc_cand)
        self.preproc_question_fn = get_preprocessing_fn(preproc_question)
        self.preproc_context_fn = get_preprocessing_fn(preproc_context)
        self.target_device = device
        self.cache_plugin = CachePlugin(cache_name="lerc", cache_kwargs=None)
        self.use_cache = use_cache
        self.verbose = verbose

    @property
    def model(self):
        if self._model is None:
            self._model = get_pretrained_lerc(device=self.target_device)
        return self._model

    def update(
        self,
        cands: list[str],
        refs: list[str],
        questions: list[str],
        contexts: Optional[list[str]] = None,
    ) -> None:
        self.cands.extend(cands)
        self.refs.extend(refs)
        self.questions.extend(questions)
        if contexts is None:
            contexts = [""] * len(cands)
        self.contexts.extend(contexts)

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
        score_list = []
        key_tuples = [
            (
                self.preproc_ref_fn(cand_raw),
                self.preproc_cand_fn(ref_raw),
                self.preproc_question_fn(question_raw),
                self.preproc_context_fn(context_raw),
            )
            for (cand_raw, ref_raw, question_raw, context_raw) in zip(
                self.cands, self.refs, self.questions, self.contexts
            )
        ]
        if self.use_cache:
            score_list_cached = self.cache_plugin.get_values(key_tuples)
        else:
            score_list_cached = [None] * len(key_tuples)

        to_save_key_tuples, to_save_scores = [], []
        n_new = 0
        for key_tuple, cached_value in zip(key_tuples, score_list_cached):
            (cand, ref, question, context) = key_tuple
            if cached_value is None:
                n_new += 1
                with torch.no_grad():
                    out_dict = self.model.predict_json(
                        dict(context=context, question=question, reference=ref, candidate=cand)
                    )
                score = out_dict["pred_score"]
                to_save_key_tuples.append(key_tuple)
                to_save_scores.append(score)
            else:
                score = cached_value

            score = max(min((score - 1) / 4, 1), 0)
            score_list.append(score)

        if self.use_cache:
            self.cache_plugin.put_values(to_save_key_tuples, to_save_scores)

        # debug print
        if self.verbose:
            print(f"Got {len(score_list)=} with {n_new=}, others cached.")
        scores = torch.tensor(score_list, device=self.device)
        if return_dict:
            return {"scores": scores}
        return scores

    def close(self):
        self._model = None


def main():
    lerc = LercMetric(verbose=True, use_cache=False)
    lerc.update(["a", "b"], ["c", "d"], ["e", "f"], ["g", "hi"])
    lerc.update(["xy"], ["zc"], ["arg"])
    print(lerc.compute())


if __name__ == "__main__":
    main()
