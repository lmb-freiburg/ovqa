from dataclasses import dataclass
from typing import List, Tuple, Optional

from transformers.utils import ModelOutput


@dataclass
class QAOutput(ModelOutput):
    """
    Args:
        answer: list of predicted answers
        top10_answers_and_probs: either list of top 10 answers and their probabilities
            or list of None
    """

    answer: List[str]
    answers: Optional[List[List[str]]] = None
    labels: Optional[List[List[str]]] = None
    top10_answers_and_probs: List[Optional[Tuple[List[str], List[float]]]] = None
