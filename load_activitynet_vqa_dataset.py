"""
Example script on how to load this dataset without depending on the entire framework.
"""

from pprint import pprint

from ovqa.paths import get_data_dir
from torch.utils.data import DataLoader

from ovqa.datasets.activitynet_vqa_dataset import ActivityNetVQADataset
from ovqa.processors import BlipImageEvalProcessor


def text_processor_noop(x):
    return x


def main():
    data_dir = get_data_dir()
    activitynet_dir = data_dir / "activitynet"
    vis_root = activitynet_dir / "frames_uncropped"
    ann_paths = [
        activitynet_dir / "activity_net.v1-3.min.json",
    ]
    vis_processor = None  # None will give a pillow image back

    # select which question the model will be asked
    question_type = "what-is-this"  # "what-is-happening-image", "what-act-is-this"

    # see ovqa/configs/datasets/activitynet.yaml
    config = {
        "question_type": question_type,
        "class_name_key": "activity",
    }
    dataset = ActivityNetVQADataset(
        vis_processor=vis_processor,
        text_processor=text_processor_noop,
        vis_root=vis_root,
        ann_paths=ann_paths,
        config=config,
    )
    datapoint = dataset[0]
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
        batch_size=2,
        collate_fn=dataset.collater,
    )
    for i, batch in enumerate(dataloader):
        image_tensor = batch.pop("image")
        print("image:", image_tensor.shape, image_tensor.dtype, image_tensor.device)
        pprint(batch)
        print()
        break


if __name__ == "__main__":
    main()
