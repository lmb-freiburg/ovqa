import math
import os
from PIL import Image
from pathlib import Path

from ovqa.datasets.classifier_vqa_dataset import ClassifierVQADataset, QUESTION_PROMPTS
from packg.iotools.jsonext import load_json


def load_activitynet_ann(ann_paths, vis_root, text_processor_fn, config, fps=16):
    # class_name_key = config.get("class_name_key", "class_name")

    # load annotations
    activitynet_json = load_json(ann_paths[0])

    # build the hierarchy
    activitynet_dict = {node["nodeId"]: node for node in activitynet_json["taxonomy"]}
    for node_id in activitynet_dict.keys():
        parentId = activitynet_dict[node_id]["parentId"]
        if parentId is None:
            continue
        if "children" not in activitynet_dict[parentId].keys():
            activitynet_dict[parentId]["children"] = []

        activitynet_dict[parentId]["children"].append(node_id)

    # extract the classes which are the leaves of the hierarchy
    classes_leafs = {}
    clsName2Id = {}
    for node in activitynet_dict.values():
        if "children" not in node.keys():
            classes_leafs[node["nodeId"]] = node
            clsName2Id[node["nodeName"]] = node["nodeId"]

    cat_ids = list(classes_leafs.keys())
    cat_ids.sort()
    cat_id_map = {v: i for i, v in enumerate(cat_ids)}
    answer_list = [
        text_processor_fn(classes_leafs[cat_ids[idx]]["nodeName"]) for idx in range(len(cat_ids))
    ]
    activity_classes = {i: classes_leafs[cat_ids[i]] for i in range(len(cat_ids))}

    # extract annotations to process
    video_annotations = activitynet_json["database"]

    video_list = list(video_annotations.keys())

    # select the frame to use for each segment
    frame_selec = config.get("frame_selec", 0.5)

    classes_set = set()
    annotations = []
    for video_name, video_dict in video_annotations.items():
        if video_dict["subset"] != "validation":
            continue
        video_frames_dir = os.path.join(vis_root, f"v_{video_name}")
        # video_frames = os.listdir(video_frames_dir)

        for segment in video_dict["annotations"]:
            classes_set.add(segment["label"])
            segment_time = segment["segment"]

            start_time = max(0.0, segment_time[0])
            end_time = min(video_dict["duration"], segment_time[1])

            # initial_frame = int(math.floor(start_time * fps))
            final_frame = max(int(math.floor(end_time * fps)), 1)
            middle_frame = min(
                max(int(math.floor((start_time + (end_time - start_time) * frame_selec) * fps)), 1),
                final_frame,
            )
            # middle_frame = max(int(math.floor((start_time + end_time) / 2 * fps)), 1)

            # add 10 zeros as padding to the frame number
            framefile = f"frame_{str(middle_frame).zfill(10)}.jpg"
            image_path = os.path.join(f"v_{video_name}", framefile)

            # check if the frame exists
            if not os.path.exists(os.path.join(video_frames_dir, framefile)):
                # get the closest frame
                video_frames = os.listdir(video_frames_dir)
                video_frame_idx = [int(frame.split(".")[0].split("_")[1]) for frame in video_frames]
                video_frame_idx.sort()
                old_frame = framefile
                middle_frame = min(video_frame_idx, key=lambda x: abs(x - middle_frame))

                # add 10 zeros as padding to the frame number
                framefile = f"frame_{str(middle_frame).zfill(10)}.jpg"
                image_path = os.path.join(f"v_{video_name}", framefile)

                # check if the frame exists
                if not os.path.exists(os.path.join(video_frames_dir, framefile)):
                    print(f"Frame {framefile} not found in {video_frames_dir}")
                    import ipdb

                    ipdb.set_trace()

                print(
                    f"Frame {old_frame} not found, took closest frame {framefile} in {video_frames_dir}"
                )

                continue
            # assert framefile in video_frames, f"Frame {framefile} not found in {video_frames_dir}"

            ann = {
                "key": len(annotations),
                "id": len(annotations),
                "video_id": video_name,
                "class_name": segment["label"],
                "category_id": clsName2Id[segment["label"]],
                "image": framefile,
                "class_idx": cat_id_map[clsName2Id[segment["label"]]],
                "segment": segment_time,
                "file_name": image_path,
            }
            annotations.append(ann)

    return annotations, answer_list, activity_classes


class ActivityNetVQADataset(ClassifierVQADataset):
    _key_field: str = "question_id"

    def __init__(
        self,
        vis_processor=None,
        text_processor=None,
        vis_root=None,
        ann_paths=None,
        config=None,
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        config: The "dataset" part of the config e.g.
            {'data_type': 'images', 'build_info': ..., 'annotations': ...,
            'type': 'eval', 'vis_processor': ..., 'text_processor': ...,
            'debug_max': 100, # <- note the -d options appear here
            }
        """
        self.config = config
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        (
            self.annotation,
            self.answer_list,
            activity_classes,
        ) = load_activitynet_ann(ann_paths, vis_root, text_processor, config)
        self.activity_classes = activity_classes
        self._add_instance_ids(key=self._key_field)
        self.return_visual = config.get("return_visual", True)

        # for clip zs classification
        self.classnames = [act_name["nodeName"] for act_name in activity_classes.values()]
        self.classsynonyms = [[act_name["nodeName"]] for act_name in activity_classes.values()]
        self.classtemplates = "none"

    def __getitem__(self, index):
        ann = self.annotation[index]

        sample = {
            "question_id": int(ann[self._key_field]),  # needed for vqa task
            "image_id": int(ann[self._key_field]),  # needed for captioning task
            "class_idx": int(ann["class_idx"]),
            "class_name": ann["class_name"],
            "image_file": ann["file_name"],
            "instance_id": int(ann[self._key_field]),  # needed for zs classification
            "label": int(ann["class_idx"]),  # needed for zs classification
        }
        if self.return_visual:
            image_path = Path(self.vis_root) / ann["file_name"]
            image_pil = Image.open(image_path).convert("RGB")
            image = self.vis_processor(image_pil)
            sample["image"] = image

        question = self.config.get("question_type", "none")
        if question not in {"none", ""}:
            sample["text_input"] = QUESTION_PROMPTS[question]

        # add object to ask followup questions about in case it is in the annotation
        if "question_followup" in ann:
            sample["text_input"] = ann["question_followup"]
        return sample
