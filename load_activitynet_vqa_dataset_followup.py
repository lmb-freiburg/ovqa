"""
Example script on how to load this dataset without depending on the entire framework.

For followup question we need existing output to ask a followup question about.
The model output for this dataset can be downloaded as described in the readme.
Also note that correctly answered questions will not be asked again, so the dataset becomes smaller.
"""
from collections import Counter
from copy import deepcopy
from pprint import pprint

from packg.debugging import connect_to_pycharm_debug_server
from packg.iotools import load_yaml
from ovqa.paths import get_data_dir
from torch.utils.data import DataLoader

from ovqa.datasets.activitynet_vqa_dataset import ActivityNetVQADataset
from ovqa.datasets.imagenet_hierarchy import load_hierarchy
from ovqa.followup import Followup
from ovqa.processors import BlipImageEvalProcessor
from ovqa.result_loader import read_single_result


def text_processor_noop(x):
    return x


def main():
    connect_to_pycharm_debug_server("edna", 33553)
    # ----- load dataset as before
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
    print(f"Original dataset length: {len(dataset)}")

    # ----- load existing model output and apply followup
    followup_cfg = load_yaml("ovqa/configs/followup/followup_activitynet.yaml")["run"]["followup_cfg"]
    pprint(followup_cfg)
    default_followup_object = followup_cfg["default_followup_object"]

    classsynonyms = dataset.classsynonyms
    synonym_dict = {name: i for i, names in enumerate(classsynonyms) for name in names}
    hier = load_hierarchy("activitynet")
    targets = {v["key"]: v["class_idx"] for v in dataset.annotation}
    follower = Followup(followup_cfg, hier, dataset.classnames, synonym_dict, targets)

    # load previous model output
    followup_prev_dir = "output/activitynet~val/blip1~ftvqa~default~none~what-is-this"
    result_obj = read_single_result(followup_prev_dir)
    assert result_obj is not None, f"Failed to read output from: {followup_prev_dir}"
    preds = result_obj.load_output()
    if next(iter(targets.keys())) not in preds:
        # fix prediction keys in this case from str '0' to int 0
        new_preds = {}
        for i, v in enumerate(dataset.annotation):
            key = v["key"]
            pred = preds[str(i)]
            new_preds[key] = pred
        preds = new_preds

    # run followup pipeline
    to_followup = follower.evaluate_pipeline(preds)
    # to_followup now looks like
    # {'val_00000003': {'status': 'followup', 'object': 'dog'},} ...
    # where status is "correct", "failed" or "followup" and in case of followup "object" is set.
    counter_followup = Counter(v["status"] for v in to_followup.values())
    print(str(dict(counter_followup)))

    # update dataset and config based on the followup questions to ask
    new_anns = []
    for ann in dataset.annotation:
        ann_followup = to_followup[ann["key"]]
        if ann_followup["status"] in "correct":
            continue
        # define the followup question
        if ann_followup["status"] == "followup":
            ask_object = ann_followup[default_followup_object]
        elif ann_followup["status"] == "failed":
            ask_object = default_followup_object
        else:
            raise ValueError(f"Unknown status: {ann_followup['status']}")
        new_ann = deepcopy(ann)
        # note this is used in ClassifierVQADataset.get_item
        new_ann["question_followup"] = ask_object
        new_anns.append(new_ann)
    dataset.annotation = new_anns
    print(f"Updated dataset, new length: {len(dataset.annotation)}")

    # ----- look at the final dataset
    # note that to get the final followup question, the text_input from the dataset must be
    # formatted with the correct prompt. the correct prompt depends on the model (see model configs)
    followup_prompt = "What type of {} is this?"

    datapoint = dataset[0]
    pprint(datapoint)
    followup_question = followup_prompt.format(datapoint["text_input"])
    print(f"Actual text_input: {followup_question}")
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
        followup_questions = [followup_prompt.format(t) for t in batch["text_input"]]
        print(f"Followup questions: {followup_questions}")
        print()
        break


if __name__ == "__main__":
    main()
