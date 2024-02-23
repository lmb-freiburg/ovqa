import math
import os

from ovqa.annotations.activitynet.activitynet_synonyms import ACTIVITYNET_CLS
from ovqa.datasets.interface_metadata import ClsMetadataInterface
from packg.iotools.jsonext import load_json
from packg.paths import get_data_dir


class ActivityNet(ClsMetadataInterface):
    """
    Statistics:
        - Around 7,652 images.
        - 200 classes.
    """

    @classmethod
    def load_split(
        cls,
        dataset_split: str = "val",
        label_field_name: str = "nodeName",
        frame_selection: str = "middle",
        fps: int = 16,
        **_kwargs,
    ):
        assert frame_selection == "middle", f"Frame selection {frame_selection} not implemented"
        available_splits = ["val"]
        assert (
            dataset_split in available_splits
        ), f"Split {dataset_split} not implemented, available: {available_splits}"

        # load json file with annotations
        json_file = cls.get_anno_file()
        activitynet_json = load_json(json_file)

        # build the hierarchy
        activitynet_dict = {node["nodeId"]: node for node in activitynet_json["taxonomy"]}
        # activity_list = [node["nodeName"] for node in activitynet_json["taxonomy"]]

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
                node["node_type"] = "leaf"
            else:
                node["node_type"] = "internal"

        cat_ids = list(classes_leafs.keys())
        cat_ids.sort()

        # answer_list = [classes_leafs[cat_ids[idx]]["nodeName"] for idx in range(len(cat_ids))]
        # activity_classes = {i: classes_leafs[cat_ids[i]] for i in range(len(cat_ids))}
        cat_id_map = {}
        classes_data = []
        for i in range(len(cat_ids)):
            cat_id_map[cat_ids[i]] = i
            clas_dict = classes_leafs[cat_ids[i]]
            clas_dict["id"] = i
            classes_data.append(clas_dict)

        # build synonym dict
        synonym_dict = {}
        synonym_all_dict = {}
        list_all_ids = list(classes_leafs.keys()) + list(
            set(activitynet_dict.keys()).difference(set(classes_leafs.keys()))
        )
        for nodeId in list_all_ids:
            nodeName = activitynet_dict[nodeId]["nodeName"]
            node_syn_dict = ACTIVITYNET_CLS[nodeName]
            activitynet_dict[nodeId]["definition"] = node_syn_dict["definition"]
            activitynet_dict[nodeId]["synonyms"] = [nodeName]
            synonym_all_dict[nodeName] = [nodeId]
            for syn in node_syn_dict["synonyms"]:
                syn = syn.lower()
                if syn not in synonym_all_dict.keys():
                    synonym_all_dict[syn] = nodeId
                    activitynet_dict[nodeId]["synonyms"].append(syn)
                if nodeId in cat_ids:
                    synonym_dict[syn] = cat_id_map[nodeId]

        # used to save the activitynet_hierarchy.json file
        # ann_file = "annotations/activitynet/simple_activitynet_hierarchy.json"
        # dump_json(activitynet_dict, ann_file, indent=2)

        # extract annotations to process
        video_annotations = activitynet_json["database"]
        vis_root = cls.get_dataset_dir()

        classes_set = set()
        annotations = {}
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
                # final_frame = int(math.floor(end_time * fps))
                middle_frame = max(int(math.floor((start_time + end_time) / 2 * fps)), 1)
                # add 10 zeros as padding to the frame number
                framefile = f"frame_{str(middle_frame).zfill(10)}.jpg"
                image_path = os.path.join(f"v_{video_name}", framefile)

                # check if the frame exists
                if not os.path.exists(os.path.join(video_frames_dir, framefile)):
                    print(f"Frame {framefile} not found in {video_frames_dir}")
                    continue

                # assert framefile in video_frames, f"Frame {framefile} not found in {video_frames_dir}"
                ann_id = len(annotations)
                ann = {
                    "datapoint_num": ann_id,
                    "video_id": video_name,
                    "class_name": segment["label"],
                    "category_id": clsName2Id[segment["label"]],
                    "image": framefile,
                    "class_idx": cat_id_map[clsName2Id[segment["label"]]],
                    "segment": segment_time,
                    "file_name": image_path,
                }
                annotations[ann_id] = ann

        assert len(annotations) > 0, f"No annotations found for activitynet {dataset_split}"
        templates_name = "none"
        return cls(
            annotations,
            classes_data,
            label_field_name,
            dataset_split=dataset_split,
            templates_name=templates_name,
            synonym_dict=synonym_dict,
        )

    @staticmethod
    def get_anno_file():
        anno_file = get_data_dir() / "activitynet/activity_net.v1-3.min.json"
        return anno_file

    @staticmethod
    def get_dataset_dir():
        return get_data_dir() / "activitynet/frames_uncropped"

    def get_image_file(self, root_id):
        image_file = self.get_dataset_dir() / self.annotations[root_id]["file_name"]
        return image_file


def main():
    dx = ActivityNet.load_split("val")
    print(len(dx.annotations))
    print(f"Done")


if __name__ == "__main__":
    main()
