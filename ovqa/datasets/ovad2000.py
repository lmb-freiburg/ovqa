from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torchvision.utils import draw_bounding_boxes

from ovqa.annotations.coco.coco_synonyms import COCO_OBJ_CLS
from ovqa.common.ovad import OVAD
from ovqa.datasets.interface_metadata import ClsMetadataInterface
from ovqa.datasets.ovad_synonyms import OVAD_ATTS_CLS
from ovqa.paths import get_ovqa_annotations_dir
from packg.iotools.jsonext import load_json
from packg.paths import get_data_dir

available_class_names = ["object", "attribute"]


class OVAD2000(ClsMetadataInterface):
    """Modified OVAD from https://ovad-benchmark.github.io/

    Statistics:
        - Around 2,000 images.
        - 80 object classes.
        - 117 attribute classes

    Reference:
        - Bravo et al. Open-vocabulary Attribute Detection. CVPR 2023.

    """

    obj_or_att = "none"

    @classmethod
    def load_split(
        cls,
        dataset_split: str = "val",
        label_field_name: str = "name",
        class_obj_att: str = "object",  # "attribute"
        category_type: str = "all",
        num_samples: int = -1,
        question_templates: list = [],
        **kwargs,
    ):
        if not isinstance(num_samples, int):
            num_samples = int(num_samples)
        if class_obj_att == "object":
            cls.obj_or_att = "object"
        elif class_obj_att == "attribute":
            cls.obj_or_att = "attribute"
        available_splits = ["val"]
        assert (
            dataset_split in available_splits
        ), f"Split {dataset_split} not implemented, available: {available_splits}"

        # load json file with annotations
        json_file = cls.get_anno_file()
        if "json_file" in kwargs.keys():
            json_file = get_data_dir() / kwargs["json_file"]

        # use ovad api
        ovad_api = OVAD(json_file)

        if class_obj_att == "object":
            cat_ids = list(ovad_api.cats.keys())
            cats = list(ovad_api.cats.values())
            # The categories in a custom json file may not be sorted.
            cat_id_map = {v: i for i, v in enumerate(cat_ids)}
            for obj_cat in cats:
                category_id = obj_cat["id"]
                obj_cat["class_idx"] = cat_id_map[category_id]
            # add synonyms to obj categories
            synonym_dict = {}
            antonym_dict = {}
            for obj_cat in cats:
                cat_syn_dict = COCO_OBJ_CLS[obj_cat["name"]]
                synonyms = list(set(cat_syn_dict["synonyms"] + [cat_syn_dict["plural"]]))
                cat_syn_dict["synonyms"] = synonyms
                obj_cat.update(
                    {
                        "meaning": cat_syn_dict["meaning"],
                        "synset": cat_syn_dict["synset"],
                        "synonyms": cat_syn_dict["synonyms"],
                        "plural": cat_syn_dict["plural"],
                    }
                )
                synonym_dict.update({syn: obj_cat["class_idx"] for syn in cat_syn_dict["synonyms"]})
            classes_data = cats

            templates_name = "openai_imagenet_template"

        elif class_obj_att == "attribute":
            classes_data = list(ovad_api.atts.values())
            cat_ids = list(ovad_api.atts.keys())
            # add synonyms to obj categories
            synonym_dict = defaultdict(list)
            antonym_dict = defaultdict(list)
            all_classes = []
            for cat_syn_dict in OVAD_ATTS_CLS:
                att_idx = cat_syn_dict["id"]
                classes_data[att_idx]["synonyms"] = cat_syn_dict["synonyms"]
                classes_data[att_idx]["antonyms"] = cat_syn_dict["antonyms"]
                classes_data[att_idx]["full_name"] = cat_syn_dict["name"]
                classes_data[att_idx]["name"] = (
                    cat_syn_dict["name"].split(":")[1].split("/")[0].strip()
                )
                for syn in cat_syn_dict["synonyms"]:
                    synonym_dict[syn].append(att_idx)
                    all_classes.append(syn)
                for ant in cat_syn_dict["antonyms"]:
                    antonym_dict[ant].append(att_idx)
                    all_classes.append(ant)
            all_classes = list(set(all_classes))
            all_classes.sort()

            templates_name = "none"
        else:
            assert False, f"Class name: {class_obj_att} not valid, available: object, attribute"

        # filter annotations based on category_type
        if category_type != "all":
            # find the level of filtering
            if class_obj_att == "object":
                supercategories = set([obj["supercategory"] for obj in classes_data])
                leafcategories = set([obj["name"] for obj in classes_data])
                if category_type in supercategories:
                    filtered_level = "supercategory"
                elif category_type in leafcategories:
                    filtered_level = "name"

                assert (
                    filtered_level != "none"
                ), f"category_type ({category_type}) needs to be a valid set of categories"

            elif class_obj_att == "attribute":
                parentcategories = set([att["parent_type"] for att in classes_data])
                typecategories = set([att["type"] for att in classes_data])
                freqcategories = set([att["freq_set"] for att in classes_data])
                leafcategories = set([att["name"] for att in classes_data])

                if category_type in typecategories:
                    filtered_level = "type"
                elif category_type in parentcategories:
                    filtered_level = "parent_type"
                elif category_type in freqcategories:
                    filtered_level = "freq_set"
                elif category_type in leafcategories:
                    filtered_level = "name"

                assert (
                    filtered_level != "none"
                ), f"category_type ({category_type}) needs to be a valid set of categories"

        # build annotations dict of object boxes
        if class_obj_att == "object":
            annotations = {}
            annotations_original = ovad_api.anns.copy()
            for i, (ann_id, ann) in enumerate(annotations_original.items()):
                # assert i == ann_id
                # add some fields required
                ann["image"] = ovad_api.imgs[ann["image_id"]]
                annotation_category_id = ann["category_id"]
                ann["class_idx"] = cat_id_map[annotation_category_id]

                if category_type != "all":
                    if category_type != classes_data[ann["class_idx"]][filtered_level]:
                        continue

                ann["datapoint_num"] = len(annotations)
                annotations[i] = ann

                if (num_samples > 0) and len(annotations) >= num_samples:
                    break

        elif class_obj_att == "attribute":
            all_prompts_templates = load_json(
                get_ovqa_annotations_dir() / "ovad/ovad_attribute_prompts.json"
            )
            # set attribute type dict
            attr_type = {}
            for att in classes_data:
                if att["type"] not in attr_type.keys():
                    attr_type[att["type"]] = set()
                attr_type[att["type"]].add(att["name"])
            attr_type = {key: list(val) for key, val in attr_type.items()}

            # write warning if no question template is found
            if len(question_templates) == 0:
                print(f"Warning: no question template found in {question_templates}")

            annotations = {}
            annotations_original = ovad_api.anns.copy()
            for i, (ann_id, ann) in enumerate(annotations_original.items()):
                # add some fields required
                ann["image"] = ovad_api.imgs[ann["image_id"]]
                ann["instance_id"] = ann_id
                pos_class_indices = []
                ann["neg_idx"] = []
                for att_idx, att_val in enumerate(ann["att_vec"]):
                    if category_type != "all":
                        if category_type != classes_data[att_idx][filtered_level]:
                            continue
                    if att_val == 1:
                        pos_class_indices.append(att_idx)
                    elif att_val == 0:
                        ann["neg_idx"].append(att_idx)

                if len(pos_class_indices) > 0:
                    # remove possible wrong negatives
                    related_pos_class_idx = []
                    for pos_idx in pos_class_indices:
                        for pos_syn in classes_data[pos_idx]["synonyms"]:
                            related_pos_class_idx.extend(synonym_dict[pos_syn])
                    neg_idx = set(ann["neg_idx"]).difference(set(related_pos_class_idx))
                    ann["neg_idx"] = list(neg_idx)

                    for pos_idx in pos_class_indices:
                        ann_to_save = ann.copy()
                        ann_to_save["class_idx"] = pos_idx
                        ann_to_save["datapoint_num"] = len(annotations)
                        ann_to_save["id"] = len(annotations)

                        # add question list to the annotation datapoint
                        obj_category = ovad_api.cats[ann_to_save["category_id"]]["name"]
                        obj_plural = COCO_OBJ_CLS[obj_category]["plural"]
                        pos_att = classes_data[ann_to_save["class_idx"]]
                        att_options_list = [att for att in attr_type[pos_att["type"]]]
                        att_options = (
                            ", ".join(att_options_list[:-1]) + " or " + att_options_list[-1]
                        )

                        att_question_list = {}
                        for prompt_type in question_templates:
                            assert prompt_type in {
                                "first_question_type",
                                "second_question_type",
                                "third_question_type",
                                "new_first_question_type",
                                "new_second_question_type",
                                "new_third_question_type",
                            }, f"Invalid question type {prompt_type}"

                            if "new_" not in prompt_type:
                                question_dict = all_prompts_templates["selected_prompts"]
                                question_dict.update(
                                    all_prompts_templates["specific_type_questions"][
                                        pos_att["type"]
                                    ]
                                )
                                question_template = question_dict[
                                    all_prompts_templates[prompt_type][pos_att["type"]]
                                ]
                            else:
                                question_template = all_prompts_templates[prompt_type][
                                    pos_att["type"]
                                ]

                            att_question = question_template.format(
                                noun=obj_category,
                                nouns=obj_plural,
                                attr_type=pos_att["type"],
                                attr=pos_att,
                                attr_options=att_options,
                            )
                            att_question_list[prompt_type] = att_question

                        if kwargs.get("prompt_brittleness", False):
                            question_dict_brittleness = all_prompts_templates["prompt_brittleness"]
                            question_templates_brittleness = list(
                                question_dict_brittleness[
                                    list(question_dict_brittleness.keys())[0]
                                ].keys()
                            )
                            for prompt_type in question_templates_brittleness:
                                question_template = question_dict_brittleness[pos_att["type"]][
                                    prompt_type
                                ]
                                att_question = question_template.format(
                                    noun=obj_category,
                                    nouns=obj_plural,
                                    attr_type=pos_att["type"],
                                    attr=pos_att,
                                    attr_options=att_options,
                                )
                                att_question_list[prompt_type] = att_question

                        ann_to_save["questions"] = att_question_list
                        annotations[ann_to_save["datapoint_num"]] = ann_to_save

                if (num_samples > 0) and len(annotations) >= num_samples:
                    break

            if "selected_datapoints" in kwargs.keys():
                print("Loading selected datapoints from {}".format(kwargs["selected_datapoints"]))
                # check file exists
                assert Path(
                    kwargs["selected_datapoints"]
                ).exists(), "File {} does not exist".format(kwargs["selected_datapoints"])
                selected_datapoints = set(load_json(kwargs["selected_datapoints"]))
                annotations = {k: v for k, v in annotations.items() if k in selected_datapoints}

        return cls(
            annotations,
            classes_data,
            label_field_name,
            dataset_split=dataset_split,
            synonym_dict=synonym_dict,
            antonym_dict=antonym_dict,
            templates_name=templates_name,
        )

    @staticmethod
    def get_anno_file():
        return get_ovqa_annotations_dir() / "ovad/ovad2000.json"

    @staticmethod
    def get_dataset_dir():
        return get_data_dir() / "coco/images/val2017"

    def get_image_file(self, index):
        image_file = self.get_dataset_dir() / self.annotations[index]["image"]["file_name"]
        return image_file

    def load_image_with_box(self, index, margin_side=0.0, min_side=0.0, square_box=False):
        ann = self.annotations[index]
        image_path = self.get_image_file(index)
        image_pil = Image.open(image_path)

        # draw box tight
        width, height = image_pil.size
        fx, fy, fw, fh = ann["bbox"]
        center_x, center_y = fx + fw / 2, fy + fh / 2
        x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh

        image_np = np.asarray(image_pil)
        avg_color = np.mean(image_np)
        # if avg_color > 128:
        #     # box_color = (0, 0, 0)
        #     box_color = (255, 0, 0)  # red
        # else:
        #     # box_color = (255, 255, 255)
        box_color = (0, 255, 0)  # green lime
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)

        image_tensor = torch.permute(torch.from_numpy(image_np), (2, 0, 1))
        image_tensor_boxes = draw_bounding_boxes(
            image_tensor,
            torch.tensor([[x1, y1, x2, y2]]),
            [""],
            [box_color],
            font_size=20,
            width=3,
        )

        # draw box input
        w_new, h_new = fw, fh
        # add margin to the box
        if margin_side > 0.0:
            w_new = min(width, w_new + margin_side * 2)
            h_new = min(height, h_new + margin_side * 2)

        # make sure the box is not too small
        bb_area = ann["area"]
        if bb_area <= min_side * min_side:
            w_new = max(min_side, w_new)
            h_new = max(min_side, h_new)

        # crop the box square
        # take biggest side and make square box around the center
        if square_box:
            max_side = min(max(w_new, h_new), min(width, height))
            w_new, h_new = max_side, max_side

        if w_new > fw or h_new > fh:
            x_new = min(max(center_x - w_new / 2, 0), width - w_new / 2)
            y_new = min(max(center_y - h_new / 2, 0), height - h_new / 2)
            nx, ny, nw, nh = int(x_new), int(y_new), int(w_new), int(h_new)
            x1, y1, x2, y2 = nx, ny, nx + nw, ny + nh

            box_color = (10, 10, 255)  # blue
            image_tensor_boxes = draw_bounding_boxes(
                image_tensor_boxes,
                torch.tensor([[x1, y1, x2, y2]]),
                ["input"],
                [box_color],
                font_size=10,
                width=1,
            )

        image = Image.fromarray(torch.permute(image_tensor_boxes, (1, 2, 0)).numpy())
        return image

    def get_datapoint_text(self, index: int):
        leaf_item = self.annotations[index].copy()
        class_info = self.classes_data[leaf_item["class_idx"]]
        leaf_item["class_info"] = class_info
        return f"Datapoint: {index} -> {leaf_item}"

    def get_datapoint_text_title(self, index: int):
        leaf_item = self.annotations[index]
        class_info = self.classes_data[leaf_item["class_idx"]]
        return class_info["name"]

    def get_scores_per_type(self, scores_per_class, name_type="parent_type"):
        assert len(scores_per_class) == len(
            self.classes_data
        ), f"Scores len {len(scores_per_class)} != class len {len(self.classes_data)}"

        type2cls = defaultdict(list)
        scores_per_type = defaultdict(float)
        # only use if results are for attibutes
        if self.obj_or_att == "attribute":
            for att_dict in self.classes_data:
                type2cls[att_dict[name_type]].append(att_dict["id"])
        else:
            for obj in self.classes_data:
                type2cls[obj[name_type]] = [obj["class_idx"]]

        for act_type, cat_idx_list in type2cls.items():
            type_scores = torch.Tensor(scores_per_class[cat_idx_list])
            scores_per_type[act_type] = type_scores.nanmean().item()

        return scores_per_type

    def get_neg_targets(self) -> Dict[str, int]:
        """
        Returns:
            Dictionary {datapoint_key (str): class_idx (int)}
        """
        return {k: v["neg_idx"] for k, v in self.annotations.items()}

    def get_questions(self) -> Dict[str, int]:
        """
        Returns:
            Dictionary {datapoint_key (str): class_idx (int)}
        """
        return {k: v["questions"] for k, v in self.annotations.items()}
