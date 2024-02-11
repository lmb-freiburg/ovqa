from collections import namedtuple, Counter

import io
from IPython.display import Image, display
from PIL import Image
from pprint import pprint

from lavis import tasks, VQATask
from ovqa.common.lavis.config import Config
from ovqa.datasets.lavis.vqav2_datasets import VQAv2EvalDataset
from visiontext.htmltools import NotebookHTMLPrinter
from packg.iotools import dumps_yaml
from visiontext.images import PILImageScaler


def load_gqa_for_notebook(dataset_name="gqa", dataset_split="balanced_testdev"):
    pr = NotebookHTMLPrinter()
    scaler = PILImageScaler(return_pillow=True)

    # build pseudo config to create datasets.
    # Note: model and most of the run config are never used, but needed to create the task
    config = {
        "datasets": {dataset_name: {"type": "eval"}},
        "run": {
            "task": "vqa",
            "test_splits": [dataset_split],
            "evaluate": True,
            "distributed": False,
            "num_beams": None,
            "max_new_tokens": None,
            "min_new_tokens": None,
            "prompt": None,
            "inference_method": None,
        },
        "model": {
            "arch": "blip2_t5",
            "model_type": "pretrain_flant5xl",
        },
    }
    config_stream = io.StringIO(dumps_yaml(config))
    cfg = Config.from_file(config_stream)
    task: VQATask = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    dataset: VQAv2EvalDataset = datasets[dataset_name][dataset_split]

    # dataset size
    ann_list = dataset.annotation
    ann_dict = {v["question_id"]: v for v in ann_list}
    num_datapoints = len(ann_list)
    print(f"Size of dataset: {num_datapoints}")

    # todo answer list

    # datapoint keys and all annotations
    print(f"---------- All annotations:")
    pprint(dict(list(ann_dict.items())[:1]))

    print()

    def get_image(t_key, t_scale=500, verbose=False):
        # assert crop is None, f"No crops defined for this dataset {dataset_name}"
        # item = meta.annotations[t_key]
        # t_class_idx = targets[t_key]
        from packg.paths import get_data_dir

        image_file = get_data_dir() / f"gqa/images/{ann_dict[t_key]['image']}"
        image = Image.open(image_file).convert("RGB")
        image_scale = scaler.scale_image_bigger_side(image, t_scale)
        return image_scale

    def fmt_answer(answer, count):
        if count == 1:
            return answer, ""
        return answer, f"(x{count})"

    font_size = 1.2

    def show_datapoint(t_key, show_text=False):
        image_scale = get_image(t_key)
        meta_item = ann_dict[t_key]
        display(image_scale)
        # t_class_idx = g_targets[t_key]
        answer = meta_item["answer"]
        full_answer = meta_item["fullAnswer"]
        if show_text:
            # todo show q and a
            answer_str = f"<b>{answer}</b> {full_answer}"
            pr.print(
                f"<span style='font-size: {font_size:.0%}'><i>{meta_item['question']}</i> "
                f"{answer_str}</span> ### data id: {t_key}"
            )
            pr.output()
        local_output_dict = {
            "question_id": t_key,
            "question": meta_item["question"],
            "answer": answer,
            "full_answer": full_answer,
            "meta_item": meta_item,
        }
        return namedtuple("VqaItem", local_output_dict.keys())(*local_output_dict.values())

    output_dict = {
        "meta": ann_dict,
        "ann_list": ann_list,
        "get_image": get_image,
        "show_datapoint": show_datapoint,
        "pr": pr,
        "scaler": scaler,
        "dataset": dataset,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
    }
    return namedtuple("VqaData", output_dict.keys())(*output_dict.values())


def main():
    load_gqa_for_notebook()
    breakpoint()
    pass


if __name__ == "__main__":
    main()
