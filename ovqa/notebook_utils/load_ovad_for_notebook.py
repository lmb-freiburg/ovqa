from collections import namedtuple
from pprint import pprint

from IPython.display import display

from ovqa.paths import get_ovqa_annotations_dir
from visiontext.htmltools import NotebookHTMLPrinter
from packg.iotools.jsonext import load_json
from ovqa.datasets.interface_metadata import ClsMetadataInterface
from ovqa.datasets.meta_loading import meta_loader
from visiontext.images import PILImageScaler

from ovqa.annotations.coco.coco_synonyms import COCO_OBJ_CLS


def load_ovad_for_notebook(
    dataset_name="ovad_attributes",
    dataset_split="val",
    question_templates=[
        "new_first_question_type",
        "new_second_question_type",
        "new_third_question_type",
    ],
):
    pr = NotebookHTMLPrinter()
    scaler = PILImageScaler(return_pillow=True)

    # class names
    meta: ClsMetadataInterface = meta_loader.load_metadata(
        dataset_name, dataset_split, question_templates=question_templates
    )
    classes_data = meta.classes_data
    print(f"---------- Class names: size {len(classes_data)}")
    print(classes_data[0])
    class_names = meta.get_class_list()
    print()

    # dataset size
    num_datapoints = len(meta.annotations)
    print(f"Size of dataset: {num_datapoints}")

    # datapoint keys and labels (class indices)
    g_targets = meta.get_targets()
    print(f"---------- Targets:")
    pprint(dict(list(g_targets.items())[:2]))
    print()

    # datapoint keys and all annotations
    print(f"---------- All annotations:")
    annotations = meta.get_annotations()
    pprint(dict(list(annotations.items())[:1]))
    print()

    # datapoint keys and labels (class indices)
    targets = meta.get_targets()
    print(f"---------- Targets: size {len(targets)}")
    pprint(dict(list(targets.items())[:5]))
    print()

    # datapoint keys and labels (class indices) of negative labels
    neg_targets = meta.get_neg_targets()
    print(f"---------- Negative Targets: size {len(neg_targets)}")
    pprint(dict(list(neg_targets.items())[:1]))
    print()

    # get object class names
    # set object classes
    objId2class = {int(val["coco_cat_id"]): key for key, val in COCO_OBJ_CLS.items()}
    print(f"---------- Object names: size {len(objId2class)}")
    pprint(dict(list(objId2class.items())[:5]))
    print()

    # set attribute type dict
    attr_type = {}
    for att in meta.classes_data:
        if att["type"] not in attr_type.keys():
            attr_type[att["type"]] = set()
        attr_type[att["type"]].add(att["name"])
    attr_type = {key: list(val) for key, val in attr_type.items()}

    # set attribute type dict
    attr_parent_type = {}
    for att in meta.classes_data:
        if att["parent_type"] not in attr_parent_type.keys():
            attr_parent_type[att["parent_type"]] = set()
        attr_parent_type[att["parent_type"]].add(att["name"])
    attr_parent_type = {key: list(val) for key, val in attr_parent_type.items()}

    # get question and predictions
    all_prompts_templates = load_json(
        get_ovqa_annotations_dir() / "ovad/ovad_attribute_prompts.json"
    )

    # load boxes
    font_size = 1.2

    def get_image(t_key, t_scale=500, verbose=False):
        image_file = meta.get_image_file(t_key)
        # load image and display
        image = meta.load_image_with_box(int(t_key), 2.0, 40.0, False)
        image_scale = scaler.scale_image_bigger_side(image, t_scale)
        return image_scale

    def get_questions(t_key=None, att_type=None, parent_type=None, new_q="new_"):
        if t_key is not None:
            data_info = get_data_info(t_key)
            return data_info["questions"]
        elif att_type is not None or parent_type is not None:
            if att_type is not None:
                att_options_list = [att for att in attr_type[att_type]]
            elif parent_type is not None:
                att_options_list = [att for att in attr_parent_type[parent_type]]
                for key_sample in range(len(annotations)):
                    ann = get_data_info(key_sample)
                    if ann["pos_att"]["parent_type"] == parent_type:
                        att_type = ann["pos_att"]["type"]

            att_options = ", ".join(att_options_list[:-1]) + " or " + att_options_list[-1]
            obj_category = "object"
            obj_plural = "objects"

            # get question
            if new_q == "":
                question_dict = all_prompts_templates["selected_prompts"]
                question_dict.update(all_prompts_templates["specific_type_questions"][att_type])

                question_template1 = question_dict[
                    all_prompts_templates["first_question_type"][att_type]
                ]
                question_template2 = question_dict[
                    all_prompts_templates["second_question_type"][att_type]
                ]
                question_template3 = question_dict[
                    all_prompts_templates["third_question_type"][att_type]
                ]
            else:
                question_dict = all_prompts_templates
                question_template1 = question_dict["new_first_question_type"][att_type]
                question_template2 = question_dict["new_second_question_type"][att_type]
                question_template3 = question_dict["new_third_question_type"][att_type]

            question1 = question_template1.format(
                noun=obj_category,
                nouns=obj_plural,
                attr_type=att_type,
                attr="",
                attr_options=att_options,
            )
            question2 = question_template2.format(
                noun=obj_category,
                nouns=obj_plural,
                attr_type=att_type,
                attr="",
                attr_options=att_options,
            )
            question3 = question_template3.format(
                noun=obj_category,
                nouns=obj_plural,
                attr_type=att_type,
                attr="",
                attr_options=att_options,
            )

            return [question1, question2, question3]

            # for key_sample in range(len(annotations)):
            #     ann = get_data_info(key_sample)
            #     if ann["pos_att"]["type"] == att_type:
            #         questions = ann["questions"]
            #         return questions

    def get_data_info(t_key):
        ann = annotations[int(t_key)]
        obj_category = objId2class[ann["category_id"]]
        obj_plural = COCO_OBJ_CLS[obj_category]["plural"]
        pos_att = meta.classes_data[ann["class_idx"]]
        neg_att = [meta.classes_data[neg_idx] for neg_idx in ann["neg_idx"]]
        att_options_list = [att for att in attr_type[pos_att["type"]]]
        att_options = ", ".join(att_options_list[:-1]) + " or " + att_options_list[-1]

        # get question
        question_dict = all_prompts_templates["selected_prompts"]
        question_dict.update(all_prompts_templates["specific_type_questions"][pos_att["type"]])

        question_template1 = question_dict[
            all_prompts_templates["first_question_type"][pos_att["type"]]
        ]
        question1 = question_template1.format(
            noun=obj_category,
            nouns=obj_plural,
            attr_type=pos_att["type"],
            attr=class_names[ann["class_idx"]],
            attr_options=att_options,
        )
        question_template2 = question_dict[
            all_prompts_templates["second_question_type"][pos_att["type"]]
        ]
        question2 = question_template2.format(
            noun=obj_category,
            nouns=obj_plural,
            attr_type=pos_att["type"],
            attr=class_names[ann["class_idx"]],
            attr_options=att_options,
        )
        question_template3 = question_dict[
            all_prompts_templates["third_question_type"][pos_att["type"]]
        ]
        question3 = question_template3.format(
            noun=obj_category,
            nouns=obj_plural,
            attr_type=pos_att["type"],
            attr=class_names[ann["class_idx"]],
            attr_options=att_options,
        )

        data_dict = {
            "ann": ann,
            "pos_att": pos_att,
            "neg_att": neg_att,
            "questions": [question1, question2, question3],
        }
        return data_dict

    def show_datapoint(t_key, show_text=False):
        image_scale = get_image(t_key)
        display(image_scale)
        t_class_idx = g_targets[t_key]
        data_info = get_data_info(t_key)
        synonyms = " / ".join(data_info["pos_att"]["synonyms"])
        questions = " / ".join(data_info["questions"])

        if show_text:
            pr.print(
                f"<span style='font-size: {font_size:.0%}'>{t_key} class {t_class_idx}: {class_names[t_class_idx]}</span>"
            )
            pr.print(f"<span style='font-size: {font_size:.0%}'>synonyms: {synonyms}</span>")
            pr.print(f"<span style='font-size: {font_size:.0%}'>questions: {questions}</span>")
            pr.output()

    output_dict = {
        "meta": meta,
        "classes_data": classes_data,
        "class_names": class_names,
        "targets": g_targets,
        "annotations": annotations,
        "get_image": get_image,
        "get_data_info": get_data_info,
        "show_datapoint": show_datapoint,
        "get_questions": get_questions,
        "pr": pr,
        "scaler": scaler,
        "num2key": meta.get_num2key(),
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "objId2class": objId2class,
    }
    return namedtuple("OVADData", output_dict.keys())(*output_dict.values())


# write a test for loading the data
# def main():
#     dl = load_ovad_for_notebook("ovad_attributes", "val")
#     dl.show_datapoint(0, show_text=True)
#     # save an example of the questions for every att type and print then all
#     questions_dict = {}
#     for key_sample in range(len(dl.annotations)):
#         ann = dl.get_data_info(key_sample)
#         if ann["pos_att"]["type"] not in questions_dict.keys():
#             questions_dict[ann["pos_att"]["type"]] = ann["questions"]

#     # print questions for every type in order
#     for type_key in sorted(questions_dict.keys()):
#         print(f"---------- {type_key}")
#         for question in questions_dict[type_key]:
#             print(question)
#         print()
