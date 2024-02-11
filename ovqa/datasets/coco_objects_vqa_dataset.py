import logging
import random
from PIL import Image
from pathlib import Path
from ovqa.common.ovad import OVAD
from packg.iotools.jsonext import load_json

from ovqa.datasets.classifier_vqa_dataset import (
    ClassifierVQADataset,
    QUESTION_PROMPTS,
)
from ovqa.annotations.coco.coco_synonyms import COCO_OBJ_CLS
from ovqa.datasets.ovad_synonyms import OVAD_ATTS_CLS


def load_ovad_ann(ann_paths, text_processor_fn, config):
    """

    Args:
        ann_paths: list length 1 with path to ovad2000.json
        text_processor_fn:
        config: must contain class_name_key, "object" for objects, "attribute" for atts
            todo must contain question_type for attributes
    Returns:

    """
    class_name_key = config.get("class_name_key", "class_name")
    category_type = config.get("category_type", "all")
    logging.info(f"Loading json {ann_paths[0]}")
    ovad_api = OVAD(ann_paths[0])
    prompts = None

    object_dict = ovad_api.cats
    attribute_dict = ovad_api.atts

    if class_name_key == "object":
        cat_ids = list(object_dict.keys())
        cats = list(object_dict.values())
        # The categories in a custom json file may not be sorted.
        # convert to lowercase for blip-like models
        answer_list = [text_processor_fn(c["name"]) for c in sorted(cats, key=lambda x: x["id"])]
        cat_id_map = {v: i for i, v in enumerate(cat_ids)}

        # add synonyms to obj categories
        for obj_cat in object_dict.values():
            cat_syn_dict = COCO_OBJ_CLS[obj_cat["name"]]
            obj_cat.update(
                {
                    "meaning": cat_syn_dict["meaning"],
                    "synset": cat_syn_dict["synset"],
                    "synonyms": cat_syn_dict["synonyms"],
                }
            )

    elif class_name_key == "attribute":
        # Make the dictionaries for evaluator of attributes
        att2idx = {}
        idx2att = {}
        attr_type = {}
        attr_parent_type = {}
        attribute_head_tail = {"head": set(), "medium": set(), "tail": set()}

        answer_list = []
        for att in attribute_dict.values():
            # get raw text synonyms
            cats_syn = att["name"].split(":")[-1].split("/")
            answer_list.append(cats_syn[0])
            cats_syn += OVAD_ATTS_CLS[att["id"]]["synonyms"]
            cats_syn = list(set(cats_syn))
            cats_syn.sort()

            cats_syn = [text_processor_fn(syn) for syn in cats_syn]
            att["synonyms"] = cats_syn

            # extract parent type and type
            att2idx[att["name"]] = att["id"]
            idx2att[att["id"]] = att["name"]

            if att["type"] not in attr_type.keys():
                attr_type[att["type"]] = set()
            attr_type[att["type"]].add(att["name"])

            if att["parent_type"] not in attr_parent_type.keys():
                attr_parent_type[att["parent_type"]] = set()
            attr_parent_type[att["parent_type"]].add(att["type"])

            attribute_head_tail[att["freq_set"]].add(att["name"])

        attr_type = {key: list(val) for key, val in attr_type.items()}
        attr_parent_type = {key: list(val) for key, val in attr_parent_type.items()}
        attribute_head_tail = {key: list(val) for key, val in attribute_head_tail.items()}

        obj_cats = object_dict
        for idx, obj in obj_cats.items():
            obj["processed_name"] = text_processor_fn(obj["name"])

        # load questions for every attribute type
        if len(ann_paths) > 1:
            prompt_type = config.get("prompt_type", "selected_prompts")
            all_prompts_templates = load_json(ann_paths[1])
            prompts_templates = all_prompts_templates[prompt_type]

    # filter per category_type selection
    if category_type != "all":
        filtered_level = "none"
        if class_name_key == "object":
            supercategories = set([obj["supercategory"] for obj in object_dict.values()])
            leafcategories = set([obj["name"] for obj in object_dict.values()])
            if category_type in supercategories:
                filtered_level = "supercategory"
            elif category_type in leafcategories:
                filtered_level = "name"

            assert (
                filtered_level != "none"
            ), f"category_type ({category_type}) needs to be a valid set of categories"

        elif class_name_key == "attribute":
            parentcategories = set([att["parent_type"] for att in attribute_dict.values()])
            typecategories = set([att["type"] for att in attribute_dict.values()])
            freqcategories = set([att["freq_set"] for att in attribute_dict.values()])
            leafcategories = set([att["name"] for att in attribute_dict.values()])

            if category_type in typecategories:
                filtered_level = "type"
            elif category_type in parentcategories:
                filtered_level = "parent_type"
            # elif category_type in typecategories:
            #     filtered_level = "type"
            elif category_type in freqcategories:
                filtered_level = "freq_set"
            elif category_type in leafcategories:
                filtered_level = "name"

            assert (
                filtered_level != "none"
            ), f"category_type ({category_type}) needs to be a valid set of categories"

    # build answer list of object boxes
    if class_name_key == "object":
        annotations = []
        for ann in ovad_api.anns.values():
            ann["image"] = ovad_api.imgs[ann["image_id"]]
            annotation_category_id = ann["category_id"]

            obj_dict = object_dict[annotation_category_id]
            # filter if needed
            if category_type != "all":
                if obj_dict[filtered_level] != category_type:
                    continue

            ann["class_idx"] = cat_id_map[annotation_category_id]
            ann["class_name"] = answer_list[ann["class_idx"]]
            annotations.append(ann)
    elif class_name_key == "attribute":
        question_type = config.get("question_type", "")
        annotations = []
        for ann in ovad_api.anns.values():
            obj = obj_cats[ann["category_id"]]
            obj_name = obj["processed_name"]

            ann["image"] = ovad_api.imgs[ann["image_id"]]
            ann["obj_name"] = obj_name
            obj_plural = COCO_OBJ_CLS[ann["obj_name"]]["plural"]

            # build questions and answers for attributes
            ann_questions = []
            ann_answers = []
            ann_class_names = []
            ann_class_idx = []
            ann_neg_class_idx = []

            # selected questions based on exploration
            if question_type in prompts_templates.keys():
                question_template = prompts_templates[question_type]
            elif prompt_type not in {
                "specific_type_questions",
                "first_question_type",
                "second_question_type",
                "third_question_type",
                "new_first_question_type",
                "new_second_question_type",
                "new_third_question_type",
            } and not isinstance(question_type, int):
                logging.warning(
                    f"The question_type ({question_type}) is not defined for prompt_type {prompt_type}. Available: "
                    + ", ".join([key for key in prompts_templates.keys()])
                )

            for att_idx, att_val in enumerate(ann["att_vec"]):
                if att_val == 0:
                    ann_neg_class_idx.append(att_idx)

                # only consider positive attributes
                elif att_val == 1:
                    attr_dict = attribute_dict[att_idx]

                    # filter if needed
                    if category_type != "all":
                        if attr_dict[filtered_level] != category_type:
                            continue

                    att_syn = attr_dict["synonyms"]
                    att_syn.sort()

                    att_options_list = [
                        att.split(":")[1].split("/")[0] for att in attr_type[attr_dict["type"]]
                    ]
                    random.shuffle(att_options_list)
                    att_options = ", ".join(att_options_list[:-1]) + " or " + att_options_list[-1]

                    if prompt_type == "specific_type_questions":
                        assert (
                            attr_dict["type"] in prompts_templates.keys()
                            and question_type in prompts_templates[attr_dict["type"]].keys()
                        ), (
                            f"The question_type ( {question_type} ) is not defined for prompt_type ( {prompt_type} ) for att type {attr_dict['type']}. Available: "
                            + ", ".join([key for key in prompts_templates.keys()])
                        )
                        question_template = prompts_templates[attr_dict["type"]][question_type]
                    elif prompt_type in {
                        "first_question_type",
                        "second_question_type",
                        "third_question_type",
                    }:
                        question_dict = all_prompts_templates["selected_prompts"]
                        question_dict.update(
                            all_prompts_templates["specific_type_questions"][attr_dict["type"]]
                        )
                        assert (
                            attr_dict["type"] in prompts_templates.keys()
                            and prompts_templates[attr_dict["type"]] in question_dict.keys()
                        ), (
                            f"The question_type ( {prompts_templates[attr_dict['type']]} ) is not defined for prompt_type ( {prompt_type} ) for att type {attr_dict['type']}. Available: "
                            + ", ".join([key for key in question_dict.keys()])
                        )
                        question_template = question_dict[prompts_templates[attr_dict["type"]]]
                    elif prompt_type == "prompt_brittleness" and isinstance(question_type, int):
                        question_dict = all_prompts_templates[prompt_type]
                        question_template = question_dict[attr_dict["type"]][str(question_type)]
                    elif prompt_type in {
                        "new_first_question_type",
                        "new_second_question_type",
                        "new_third_question_type",
                    }:
                        question_dict = all_prompts_templates[prompt_type]
                        assert (
                            attr_dict["type"] in question_dict.keys()
                        ), f"attr type {attr_dict['type']} not in question_dict keys {question_dict.keys()} for prompt_type {prompt_type}"
                        question_template = question_dict[attr_dict["type"]]
                    else:
                        assert (
                            question_template
                        ), f"question_template is not defined. Define question_type ({question_type}) and prompt_type ({prompt_type})"

                    att_question = question_template.format(
                        noun=obj_name,
                        nouns=obj_plural,
                        attr_type=attr_dict["type"],
                        attr=att_syn[0],
                        attr_options=att_options,
                    )

                    ann_questions.append(att_question)
                    # text_processor_fn(att_question))
                    ann_answers.append(text_processor_fn(att_syn[0]))
                    ann_class_names.append(attr_dict["name"])
                    ann_class_idx.append(att_idx)

            if len(ann_class_idx) > 0:
                for pos_idx in range(len(ann_class_idx)):
                    ann_to_save = ann.copy()
                    ann_to_save.update(
                        {
                            "id": len(annotations),
                            "key": len(annotations),
                            "ovad_id": ann["id"],
                            "class_idx": ann_class_idx[pos_idx],
                            "class_name": ann_class_names[pos_idx],
                            "questions": ann_questions[pos_idx],
                            "answers": ann_answers[pos_idx],
                            "neg_class_idx": ann_neg_class_idx,
                        }
                    )
                    annotations.append(ann_to_save)
                # ann["class_idx"] = ann_class_idx
                # ann["class_name"] = ann_class_names

                # ann["questions"] = ann_questions
                # ann["answers"] = ann_answers

                # ann["neg_class_idx"] = ann_neg_class_idx
                # annotations.append(ann)

    return annotations, answer_list, prompts, object_dict, attribute_dict


def get_cropped_image(original_image, bbox_ann, margin_side, min_side, square_box):
    # crop box
    width, height = original_image.size
    x, y, w, h = bbox_ann["bbox"]
    center_x, center_y = x + w / 2, y + h / 2
    x, y, w, h = int(x), int(y), int(w), int(h)
    bb_area = bbox_ann["area"]
    w_new, h_new = w, h

    # add margin to the box
    if margin_side > 0.0:
        w_new = min(width, w_new + margin_side * 2)
        h_new = min(height, h_new + margin_side * 2)

    # make sure the box is not too small
    if bb_area <= min_side * min_side:
        w_new = max(min_side, w_new)
        h_new = max(min_side, h_new)

    # crop the box square
    # take biggest side and make square box around the center
    if square_box:
        max_side = min(max(w_new, h_new), min(width, height))
        w_new, h_new = max_side, max_side

    if w_new > w or h_new > h:
        x_new = min(max(center_x - w_new / 2, 0), width - w_new / 2)
        y_new = min(max(center_y - h_new / 2, 0), height - h_new / 2)
        x, y, w, h = int(x_new), int(y_new), int(w_new), int(h_new)

    cropped_image = original_image.crop((x, y, x + w, y + h))
    return cropped_image


def debug_loading(debug_dir, original_image, crop_image, sample, annotation):
    debug_dir = Path()
    debug_dir.mkdir(parents=True, exist_ok=True)
    print(annotation["id"], annotation["class_name"])
    print("original_image", original_image.size)
    print("crop_image", crop_image.size)

    original_image.save(debug_dir / f"{str(annotation['image_id']).zfill(5)}.jpg")
    crop_image.save(
        debug_dir / f"{str(annotation['image_id']).zfill(5)}_{str(annotation['id']).zfill(5)}.jpg"
    )

    if annotation["id"] > 100:
        import ipdb

        ipdb.set_trace()


class COCOObjectsVQADataset(ClassifierVQADataset):
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
            self.prompts,
            obj_classes,
            att_classes,
        ) = load_ovad_ann(ann_paths, text_processor, config)
        self.obj_classes = obj_classes
        self.att_classes = att_classes
        self._add_instance_ids(key=self._key_field)
        self.return_visual = config.get("return_visual", True)

        # for clip zs classification
        self.classnames = [obj_name["name"] for obj_name in obj_classes.values()]
        self.classsynonyms = [obj_name["synonyms"] for obj_name in obj_classes.values()]
        self.classtemplates = "openai_imagenet_template"

    def __getitem__(self, index):
        ann = self.annotation[index]

        sample = {
            "question_id": int(ann[self._key_field]),  # needed for vqa task
            "image_id": int(ann[self._key_field]),  # needed for captioning task
            "class_idx": int(ann["class_idx"]),
            "class_name": ann["class_name"],
            "image_file": ann["image"]["file_name"],
            "instance_id": int(ann[self._key_field]),  # needed for zs classification
            "label": int(ann["class_idx"]),  # needed for zs classification
        }
        if self.return_visual:
            image_path = Path(self.vis_root) / ann["image"]["file_name"]
            image_pil = Image.open(image_path).convert("RGB")
            original_image = image_pil.copy()
            image_pil = get_cropped_image(
                original_image,
                ann,
                self.config.get("margin_side", 0.0),
                self.config.get("min_side", 40.0),
                self.config.get("square_box", False),
            )
            image = self.vis_processor(image_pil)
            sample["image"] = image

        question = self.config.get("question_type", "none")
        if question not in {"none", ""}:
            sample["text_input"] = QUESTION_PROMPTS[question]

        # debug_loading(self.config.get("debug_dir", ""), original_image, crop_image, sample, annotation)
        return sample


class OVADAttributesVQADataset(ClassifierVQADataset):
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
            self.prompts,
            obj_classes,
            att_classes,
        ) = load_ovad_ann(ann_paths, text_processor, config)
        self.obj_classes = obj_classes
        self.att_classes = att_classes

        self._add_instance_ids(key=self._key_field)
        self.return_visual = config.get("return_visual", True)

        # for clip zs classification
        self.classnames = [att_name["name"] for att_name in att_classes.values()]
        self.classsynonyms = [att_name["synonyms"] for att_name in att_classes.values()]
        self.classtemplates = "none"

    def __getitem__(self, index):
        ann = self.annotation[index]

        sample = {
            "question_id": int(ann[self._key_field]),  # needed for vqa task
            "instance_id": int(ann[self._key_field]),  # needed for zs classification
            "image_id": int(ann[self._key_field]),  # needed for captioning task
            "class_idx": ann["class_idx"],
            "class_name": ann["class_name"],
            "image_file": ann["image"]["file_name"],
            "text_input": ann["questions"],
            "answers": ann["answers"],
            "neg_class_idx": ann["neg_class_idx"],
            # "question_id": int(ann[self._key_field]),  # needed for vqa task
            # "image_id": int(ann[self._key_field]),  # needed for captioning task
            # "class_idx": int(ann["class_idx"]),
            # "class_name": ann["class_name"],
            # "image_file": ann["file_name"],
            # "instance_id": int(ann[self._key_field]),  # needed for zs classification
            # "label": int(ann["class_idx"]),  # needed for zs classification
        }
        if self.return_visual:
            image_path = Path(self.vis_root) / ann["image"]["file_name"]
            image_pil = Image.open(image_path).convert("RGB")
            original_image = image_pil.copy()
            image_pil = get_cropped_image(
                original_image,
                ann,
                self.config.get("margin_side", 0.0),
                self.config.get("min_side", 40.0),
                self.config.get("square_box", False),
            )
            image = self.vis_processor(image_pil)
            sample["image"] = image

        # debug_loading(self.config.get("debug_dir", ""), original_image, crop_image, sample, annotation)
        return sample
