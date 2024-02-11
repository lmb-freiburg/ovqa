from typing import List

from ovqa.metrics.clip_match import EmbeddingAccuracy
from ovqa.metrics.simple import (
    compare_is_equal,
    compare_is_contained,
    check_length_of_cand,
    TextComparison,
)
from ovqa.metrics.torchmetrics_ext import MetricCollectionExt
from ovqa.textutils.embeddings import EmbeddingsPackageConst


def setup_text_metrics() -> MetricCollectionExt:
    metrics = {
        "Cont": TextComparison(comparison_fn=compare_is_contained),
        "EM": TextComparison(comparison_fn=compare_is_equal),
        "Len": TextComparison(comparison_fn=check_length_of_cand, format_str="{:.2f}"),
    }
    return MetricCollectionExt(metrics, compute_groups=False)


def setup_clipmatch_metrics(
    class_names: List[str], templates_name: str = "openai_imagenet_template"
) -> MetricCollectionExt:
    metrics = {
        # Top-K Accuracy using ClipMatch with EVA-CLIP-G
        "ClipM1": EmbeddingAccuracy(
            class_names,
            top_k=1,
            package_name=EmbeddingsPackageConst.OPEN_CLIP,
            embedder_name="EVA01-g-14/laion400m_s11b_b41k",
            templates_name=templates_name,
        ),
        "ClipM5": EmbeddingAccuracy(
            class_names,
            top_k=5,
            package_name=EmbeddingsPackageConst.OPEN_CLIP,
            embedder_name="EVA01-g-14/laion400m_s11b_b41k",
            templates_name=templates_name,
        ),
    }
    return MetricCollectionExt(metrics, compute_groups=False)
