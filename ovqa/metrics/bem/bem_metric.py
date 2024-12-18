import torch
from tqdm import tqdm

from ovqa.paths import get_from_environ
from visiontext.cacheutils import CachePlugin
from ovqa.metrics.preprocessing import PrepC, get_preprocessing_fn
from ovqa.metrics.bem.bem_query import query_bem
from ovqa.metrics.torchmetrics_ext import MetricExt


class BemQueryMetric(MetricExt):
    def __init__(
        self,
        server=None,
        preproc_cand=PrepC.NONE,
        preproc_ref=PrepC.NONE,
        preproc_question=PrepC.NONE,
        use_cache: bool = False,
        batch_size: int = 64,  # especially for large inputs from llava, bem can run oom on 24GB
        verbose: bool = True,
    ):
        super().__init__()
        if server is None:
            print(f"Trying to find server from env variable BEM_SERVER")
            try:
                server = get_from_environ("BEM_SERVER")
                print(f"Got server from env variable BEM_SERVER: {server}")
            except KeyError:
                server = "localhost:5000"
                print(f"Not found, using default server: {server}")

        self.add_state("cands", default=[], dist_reduce_fx="cat")
        self.add_state("refs", default=[], dist_reduce_fx="cat")
        self.add_state("questions", default=[], dist_reduce_fx="cat")
        self.preproc_cand_fn = get_preprocessing_fn(preproc_ref)
        self.preproc_ref_fn = get_preprocessing_fn(preproc_cand)
        self.preproc_question_fn = get_preprocessing_fn(preproc_question)

        self.server = server
        self.cache_plugin = CachePlugin(cache_name="bem", cache_kwargs=None)
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.verbose = verbose

    def update(
        self,
        cands: list[str],
        refs: list[str],
        questions: list[str],
        *args: list[str],
    ) -> None:
        self.cands.extend(cands)
        self.refs.extend(refs)
        self.questions.extend(questions)

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
        cands_raw = self.cands
        cands = [self.preproc_cand_fn(cand_raw) for cand_raw in cands_raw]
        refs_raw = self.refs
        refs = [self.preproc_ref_fn(ref_raw) for ref_raw in refs_raw]
        questions_raw = self.questions
        questions = [self.preproc_question_fn(question_raw) for question_raw in questions_raw]

        input_tuples = list(zip(cands, refs, questions))
        if self.use_cache:
            score_list = self.cache_plugin.get_values(input_tuples)
        else:
            score_list = [None] * len(input_tuples)

        to_run = []
        for i, (input_tuple, score) in enumerate(zip(input_tuples, score_list)):
            if score is None:
                to_run.append((i, input_tuple))

        batch, n_computed = [], 0
        for i, input_tuple in tqdm(to_run, disable=not self.verbose, desc="BEM"):
            batch.append((i, input_tuple))
            if len(batch) >= self.batch_size or i == len(to_run) - 1:
                batch_ids = [tup[0] for tup in batch]
                batch_data = [tup[1] for tup in batch]
                bem_formatted_data = [
                    {
                        "candidate": cand,
                        "reference": ref,
                        "question": question,
                    }
                    for cand, ref, question in batch_data
                ]
                n_computed += len(batch)
                batch_score_list = query_bem(bem_formatted_data, server=self.server)

                for j, score in zip(batch_ids, batch_score_list):
                    assert score_list[j] is None, "This should not happen"
                    score_list[j] = score
                batch = []
                if self.use_cache:
                    self.cache_plugin.put_values(batch_data, batch_score_list)

        if self.verbose:
            print(f"BEM: Computed {n_computed} of {len(score_list)}")
        scores = torch.tensor(score_list, device=self.device)
        if return_dict:
            return {"scores": scores}
        return scores

    def close(self):
        pass


def main():
    metric = BemQueryMetric()
    questions = [
        "why is the sky blue",
        "how do plants make food",
        "what is the capital of France",
        "who wrote 'Romeo and Juliet'",
        "why do we sneeze",
        "what is photosynthesis",
    ]

    refs = [
        "light scattering",
        "photosynthesis",
        "Paris",
        "William Shakespeare",
        "to clear out irritants from the nose",
        "process by which plants convert light energy to chemical energy",
    ]

    cand = [
        "scattering of light",
        "conversion of sunlight into glucose",
        "Paris city",
        "Shakespeare",
        "we sneeze if our foot hurts",
        "conversion of light to energy in plants",
    ]
    metric.reset()
    metric.update(cand, refs, questions)
    scores = metric.compute_per_datapoint()

    for i in range(len(scores)):
        print(f"Question: {questions[i]}")
        print(f"Reference: {refs[i]}")
        print(f"Candidate: {cand[i]}")
        print(f"Score: {scores[i]}")
        print()

    print()
    print(f"===================== with ?")
    print()
    questions = [f"{q}?" for q in questions]
    metric.reset()
    metric.update(cand, refs, questions)
    scores = metric.compute_per_datapoint()

    for i in range(len(scores)):
        print(f"Question: {questions[i]}")
        print(f"Reference: {refs[i]}")
        print(f"Candidate: {cand[i]}")
        print(f"Score: {scores[i]}")
        print()


if __name__ == "__main__":
    main()
