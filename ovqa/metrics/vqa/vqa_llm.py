from __future__ import annotations

from spacy.tokens.span_group import deepcopy

from ovqa.metrics.llm.llama_incontext_kvcache import LlamaMetricKV
from ovqa.metrics.vqa.base_vqa_metric import aggregate_vqa_scores


class VqaLlmMetric(LlamaMetricKV):
    """
    Cands, refs, questions will be added
    Just with the differences that refs is list of list, instead of list

    """

    def __init__(self, hit_value=0.3):
        super().__init__()
        self.hit_value = hit_value

    def compute_per_answer(self) -> list[list[float]]:
        # save original data
        orig_questions = deepcopy(self.questions)
        orig_cands = deepcopy(self.cands)
        orig_refs = deepcopy(self.refs)
        print(f"Got input {len(orig_cands)} questions {sum(len(v) for v in orig_refs)} answers")

        # deduplicate pairs
        texts = {}
        for question, pred, answer_list in zip(self.questions, self.cands, self.refs):
            for answer in answer_list:
                texts[(question, pred, answer)] = None
        print(f"Got {len(texts)} deduplicated pairs")

        # print(orig_questions, orig_cands, orig_refs)
        # print(texts)

        # extract new values from dict keys of texts
        self.cands = []
        self.refs = []
        self.questions = []
        for i, (question, cand, ref) in enumerate(list(texts.keys())):
            self.cands.append(cand)
            self.refs.append(ref)
            self.questions.append(question)
            texts[(question, cand, ref)] = i

        # now run the computation on this data
        acc_tensor = self.compute_per_datapoint().tolist()

        # restore original data
        out_list = []
        for question, pred, answer_list in zip(orig_questions, orig_cands, orig_refs):
            out_list_single = []
            for answer in answer_list:
                i = texts[(question, pred, answer)]
                acc = acc_tensor[i]
                out_list_single.append(acc)
            out_list.append(out_list_single)

        return out_list

    def aggregate_vqa_scores(self, scores_per_answer):
        return aggregate_vqa_scores(scores_per_answer, hit_value=self.hit_value)

    def close(self):
        pass
