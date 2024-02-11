from __future__ import annotations

import re

from ovqa.common.lavis.vqa_tools.vqa_eval import VQAEval
from packg import Const
from visiontext.nlp.lemmatizer import get_lemmatizer


class PrepC(Const):
    VQA_PRED = "vqa_pred"  # official coco vqa preprocessing for prediction
    VQA_ANSWER = "vqa_answer"  # official coco vqa preprocessing for answer
    SIMPLE = "simple"  # remove non alphanum, lowercase, strip duplicate whitespaces
    NONE = "none"
    LEMMATIZE = "lemmatize"


def create_chained_function(fns):
    def single_function(x):
        for fn in fns:
            x = fn(x)
        return x

    return single_function


def get_preprocessing_fn(preprocessing_name: str | list[str]):
    if isinstance(preprocessing_name, list):
        # apply multiple preprocessing functions in order
        fns = [get_preprocessing_fn(name) for name in preprocessing_name]
        return create_chained_function(fns)

    if preprocessing_name == PrepC.VQA_PRED:
        return preprocess_vqa_prediction
    if preprocessing_name == PrepC.VQA_ANSWER:
        return preprocess_vqa_answer
    if preprocessing_name == PrepC.SIMPLE:
        return preprocess_text_simple
    if preprocessing_name == PrepC.NONE:
        return noop
    if preprocessing_name == PrepC.LEMMATIZE:
        # todo batch lemmatization might be faster
        lemmatizer = get_lemmatizer()
        return lemmatizer.lemmatize

    raise ValueError(
        f"Unknown preprocessing fn '{preprocessing_name}', should be one of "
        f"{list(PrepC.values())}"
    )


vqa_eval = VQAEval()


def preprocess_vqa_prediction(pred: str) -> str:
    pred = pred.replace("\n", " ")
    pred = pred.replace("\t", " ")
    pred = pred.strip()
    pred = vqa_eval.processPunctuation(pred)
    pred = vqa_eval.processDigitArticle(pred)
    return pred


def preprocess_vqa_answer(answer: str) -> str:
    answer = vqa_eval.processPunctuation(answer)
    return answer


RE_ALNUM = re.compile(r"[^a-zA-Z0-9 ]+")


def preprocess_text_simple(in_str: str) -> str:
    """Remove non-alphanumeric characters, lowercase, and strip duplicate whitespaces."""
    return " ".join(RE_ALNUM.sub(" ", in_str).strip().lower().split())


def noop(in_str: str) -> str:
    return in_str
