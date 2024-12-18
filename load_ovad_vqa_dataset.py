"""
Example script on how to load this dataset without depending on the entire framework.
"""

from pprint import pprint

from ovqa.paths import get_data_dir
from torch.utils.data import DataLoader

from ovqa.datasets.coco_objects_vqa_dataset import OVADAttributesVQADataset
from ovqa.processors import BlipImageEvalProcessor


def text_processor_noop(x):
    return x


def main():
    data_dir = get_data_dir()
    coco_dir = data_dir / "coco"
    vis_root = coco_dir / "images" / "val2017"
    ann_paths = [
        "ovqa/annotations/ovad/ovad2000.json",
        "ovqa/annotations/ovad/ovad_attribute_prompts.json",
    ]
    vis_processor = None  # None will give a pillow image back

    # select which question the model will be asked
    # options: "new_first_question_type", "new_second_question_type", "new_third_question_type"
    question_type = "new_first_question_type"

    # see ovqa/configs/datasets/coco.yaml
    config = {
        "class_name_key": "attribute",
        "square_box": False,
        "min_side": 40.0,
        "margin_side": 2.0,
        "prompt_type": question_type,
        "category_type": "all",
    }
    dataset = OVADAttributesVQADataset(
        vis_processor=vis_processor,
        text_processor=text_processor_noop,
        vis_root=vis_root,
        ann_paths=ann_paths,
        config=config,
    )
    datapoint = dataset[0]
    print("neg_class_idx", datapoint.pop("neg_class_idx"))
    pprint(datapoint)
    print()

    # in order to use a dataloader, we need to transform the images to tensors, so we can stack them
    dataset.vis_processor = BlipImageEvalProcessor(
        image_size=224, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=0,
        batch_size=16,
        collate_fn=dataset.collater,
    )
    for i, batch in enumerate(dataloader):
        image_tensor = batch.pop("image")
        print("image:", image_tensor.shape, image_tensor.dtype, image_tensor.device)
        print("neg_class_idx", batch.pop("neg_class_idx"))
        pprint(batch)
        print()
        break


if __name__ == "__main__":
    main()
