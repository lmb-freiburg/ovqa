from collections import namedtuple
from pprint import pprint

from IPython.display import display

from visiontext.htmltools import NotebookHTMLPrinter
from ovqa.datasets.interface_metadata import ClsMetadataInterface
from ovqa.datasets.meta_loading import meta_loader
from visiontext.images import PILImageScaler

from ovqa.annotations.coco.coco_synonyms import COCO_OBJ_CLS


def load_coco_for_notebook(
    dataset_name="coco",
    dataset_split="val",
):
    pr = NotebookHTMLPrinter()
    scaler = PILImageScaler(return_pillow=True)

    # class names
    meta: ClsMetadataInterface = meta_loader.load_metadata(dataset_name, dataset_split)
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

    # get object class names
    # set object classes
    objId2class = {int(val["coco_cat_id"]): key for key, val in COCO_OBJ_CLS.items()}
    print(f"---------- Object names: size {len(objId2class)}")
    pprint(dict(list(objId2class.items())[:5]))
    print()

    # load boxes
    font_size = 1.2

    def get_image(t_key, t_scale=500, verbose=False):
        image_file = meta.get_image_file(t_key)
        # load image and display
        image = meta.load_image_with_box(int(t_key), 2.0, 40.0, False)
        image_scale = scaler.scale_image_bigger_side(image, t_scale)
        return image_scale

    def show_datapoint(t_key, show_text=False):
        image_scale = get_image(t_key)
        display(image_scale)
        t_class_idx = g_targets[t_key]
        ann = annotations[int(t_key)]
        obj_category = meta.classes_data[ann["class_idx"]]
        image_dict = ann["image"]
        if show_text:
            pr.print(
                f"<span style='font-size: {font_size:.0%}'>{t_key} class {t_class_idx}: {class_names[t_class_idx]}</span>"
            )
            pr.output()

    output_dict = {
        "meta": meta,
        "classes_data": classes_data,
        "class_names": class_names,
        "targets": g_targets,
        "annotations": annotations,
        "get_image": get_image,
        "show_datapoint": show_datapoint,
        "pr": pr,
        "scaler": scaler,
        "num2key": meta.get_num2key(),
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "objId2class": objId2class,
    }
    return namedtuple("COCOData", output_dict.keys())(*output_dict.values())


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
