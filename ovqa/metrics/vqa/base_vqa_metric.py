import torch

from ovqa.metrics.torchmetrics_ext import MetricExt


# noinspection PyAbstractClass
class BaseVqaMetric(MetricExt):
    def __init__(self):
        super().__init__()
        self.add_state("pred_values", default=[], dist_reduce_fx="cat")
        self.add_state("pred_keys", default=[], dist_reduce_fx="cat")
        self.add_state("answers", default=[], dist_reduce_fx="cat")

    def update(self, values: list[str], answers: list[list[str]], *args) -> None:
        """Update state with data.

        Args:
            values: Prediction values (model output).
            answers: Ground truth answers.
            *args: Ignored arguments (e.g. questions)
        """
        self.pred_values.extend(values)
        self.answers.extend(answers)

    def compute(self) -> float:
        """Compute the metric."""
        acc_list = self.compute_per_datapoint()
        acc = torch.mean(acc_list)
        return acc.item()

    def aggregate_vqa_scores(self, scores_per_answer):
        """"Turn per-answer scores into per-datapoint scores"""
        return aggregate_vqa_scores(scores_per_answer, hit_value=self.hit_value)


def aggregate_vqa_scores(
    scores_per_answer: list[list[float]], hit_value: float = 0.3
) -> list[float]:
    """
    Turn per-answer scores into per-datapoint scores for datasets with multiple
    answers per question.

    Note: Could be optimized by using torch instead of python lists
    """
    acc_list = []
    for answer_scores in scores_per_answer:
        total_score = sum(answer_scores)
        acc = min(1, total_score * hit_value)
        acc_list.append(acc)
    return acc_list
