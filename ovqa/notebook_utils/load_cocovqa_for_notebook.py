from collections import namedtuple, Counter

from IPython.display import Image, display
from PIL import Image
from pprint import pprint

from ovqa.datasets.load_dataset import load_lavis_dataset
from visiontext.htmltools import NotebookHTMLPrinter
from visiontext.images import PILImageScaler


def load_cocovqa_for_notebook(dataset_name="coco_vqa", dataset_split="minival"):
    pr = NotebookHTMLPrinter()
    scaler = PILImageScaler(return_pillow=True)

    dataset = load_lavis_dataset(dataset_name, dataset_split, "vqa")

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
        from ovqa.paths import get_data_dir

        image_file = get_data_dir() / f"coco/images/{ann_dict[t_key]['image']}"
        image = Image.open(image_file).convert("RGB")
        if t_scale is not None:
            image = scaler.scale_image_bigger_side(image, t_scale)
        return image

    def fmt_answer(answer, count):
        if count == 1:
            return answer, ""
        return answer, f"(x{count})"

    font_size = 1.2

    def show_datapoint(t_key, show_text=False, t_scale=500):
        image_scale = get_image(t_key, t_scale=t_scale)
        meta_item = ann_dict[t_key]
        display(image_scale)
        # t_class_idx = g_targets[t_key]
        answer_list = meta_item["answer"]
        answers_and_counts = Counter(answer_list).most_common()
        if show_text:
            # todo show q and a
            answer_strs = []
            for answer, count in answers_and_counts:
                fmted_answers, fmted_count = fmt_answer(answer, count)
                answer_strs.append(f"<b>{fmted_answers}</b> {fmted_count}")
            answer_str = f" | ".join(answer_strs)
            pr.print(
                f"<span style='font-size: {font_size:.0%}'><i>{meta_item['question']}</i> "
                f"{answer_str}</span> ### data id: {t_key}"
            )
            pr.output()
        local_output_dict = {
            "question_id": t_key,
            "question": meta_item["question"],
            "answer_list": answer_list,
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
