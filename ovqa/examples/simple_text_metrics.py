"""
Examples for Contains, ExactMatch and F1 metrics for text.
"""

from ovqa.metrics.preprocessing import PrepC
from ovqa.metrics.simple import (
    get_metric_is_equal_vqa_prep,
    get_metric_is_equal_default_prep,
    get_metric_is_cont_default_prep,
    get_metric_is_cont_vqa_prep,
    get_f1_score_default_prep,
    TextComparisonSynonyms,
    compare_is_equal,
)


def main():
    labels = "black cat", "white dog", "black dog", "5 lakes", "skateboard"
    captions = (
        "an image of a black cat",
        "a dog near a black house",
        "a dog near a black house",
        "the 7 seas",
        "skateboard",
    )
    for metric_fn in [
        get_metric_is_equal_vqa_prep,
        get_metric_is_cont_vqa_prep,
        get_metric_is_equal_default_prep,
        get_metric_is_cont_default_prep,
        get_f1_score_default_prep,
    ]:
        print(metric_fn.__name__)
        metric = metric_fn()
        metric.reset()
        metric.update(captions, labels)
        print(f"    Average: {metric.format(metric.compute())}")

        for label, caption in zip(labels, captions):
            metric.reset()
            metric.update([caption], [label])
            print(f"    {metric.format(metric.compute())} for GT {label} REF {caption}")

    # example for synonym metric
    print(f"EMSyn example")
    metric = TextComparisonSynonyms(
        comparison_fn=compare_is_equal,
        preproc_cand=PrepC.SIMPLE,
        preproc_ref=PrepC.SIMPLE,
        name2syn={"big": ["large", "huge"]},
    )
    labels = "big", "big", "big", "big"
    captions = "large", "huge", "big", "small"
    for label, caption in zip(labels, captions):
        metric.reset()
        metric.update([caption], [label])
        print(f"    {metric.format(metric.compute())} for GT {label} REF {caption}")


if __name__ == "__main__":
    main()
