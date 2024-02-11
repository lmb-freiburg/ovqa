"""
View results of VQA tasks.

python view_results_vqa.py -s minival
"""
from ovqa.metrics.llm.llama_incontext_kvcache import LlamaMetricKV
from ovqa.metrics.preprocessing import PrepC
from ovqa.metrics.simple import (
    get_metric_is_equal_default_prep,
    get_metric_is_equal_vqa_prep,
    get_metric_is_cont_vqa_prep,
    get_metric_is_cont_default_prep,
    TextComparison,
    check_length_of_cand,
    compare_is_equal,
    compare_is_contained,
)
from ovqa.metrics.torchmetrics_ext import MetricCollectionExt
from ovqa.metrics.vqa.vqa_llm import VqaLlmMetric
from ovqa.metrics.vqa.vqa_predlen import VqaPredLen
from ovqa.metrics.vqa.vqa_text_comparison import VqaTextComparison


def get_singleanswer_vqa_metrics(_answer_list=None, llm=False):
    metrics_dict = {
        "EM": get_metric_is_equal_vqa_prep(),
        "EM_S": get_metric_is_equal_default_prep(),
        "Cont": get_metric_is_cont_vqa_prep(),
        "Cont_S": get_metric_is_cont_default_prep(),
        "length": TextComparison(
            comparison_fn=check_length_of_cand,
            preproc_cand=PrepC.VQA_PRED,
            preproc_ref=PrepC.VQA_PRED,
            format_str="{:.2f}",
        ),
        "length_S": TextComparison(
            comparison_fn=check_length_of_cand,
            preproc_cand=PrepC.SIMPLE,
            preproc_ref=PrepC.SIMPLE,
            format_str="{:.2f}",
        ),
    }
    if llm:
        metrics_dict["llama2_70b_5c"] = LlamaMetricKV()
    return MetricCollectionExt(metrics_dict, compute_groups=False)


def get_vqav2_metrics(_answer_list=None, llm=False):
    """
    Note: VQAv2 ground truth answers are heavily preprocessed, therefore PrepC.VQA_ANSWER is
        sufficient. For datasets with more "raw" answers, another preprocessing function
        (like PrepC.VQA_PRED or PrepC.SIMPLE) might be better.

    """
    metrics_dict = {
        "EM": VqaTextComparison(
            comparison_fn=compare_is_equal,
            preproc_pred=PrepC.VQA_PRED,
            preproc_target=PrepC.VQA_ANSWER,
        ),
        "Cont": VqaTextComparison(
            comparison_fn=compare_is_contained,
            preproc_pred=PrepC.VQA_PRED,
            preproc_target=PrepC.VQA_ANSWER,
        ),
    }
    if llm:
        metrics_dict["llama2_70b_5c"] = VqaLlmMetric()
    metrics_dict["length"] = VqaPredLen(preproc_pred=PrepC.VQA_PRED, preproc_target=PrepC.VQA_PRED)
    metrics = MetricCollectionExt(metrics_dict, compute_groups=False)
    return metrics
