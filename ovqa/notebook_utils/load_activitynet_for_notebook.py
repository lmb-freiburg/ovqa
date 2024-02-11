from collections import namedtuple
from pprint import pprint

from IPython.display import Image, display
from PIL import Image

from visiontext.htmltools import NotebookHTMLPrinter
from packg.iotools.jsonext import load_json
from ovqa.paths import get_ovqa_annotations_dir
from ovqa.datasets.interface_metadata import ClsMetadataInterface
from ovqa.datasets.meta_loading import meta_loader
from visiontext.images import PILImageScaler
from ovqa.datasets.imagenet_hierarchy import load_hierarchy


def load_activitynet_for_notebook(dataset_name="activitynet", dataset_split="val"):
    pr = NotebookHTMLPrinter()
    scaler = PILImageScaler(return_pillow=True)

    meta: ClsMetadataInterface = meta_loader.load_metadata(dataset_name, dataset_split)
    classes_data = meta.classes_data
    print(classes_data[0])

    # dataset size
    num_datapoints = len(meta.annotations)
    print(f"Size of dataset: {num_datapoints}")

    # class names
    class_names = meta.get_class_list()
    print(f"---------- Class names: size {len(class_names)}")
    print(class_names[:5], "...")
    # print(class_names)
    print()

    # datapoint keys and labels (class indices)
    g_targets = meta.get_targets()
    print(f"---------- Targets:")
    pprint(dict(list(g_targets.items())[:2]))
    print()

    # datapoint keys and all annotations
    print(f"---------- All annotations:")
    ann = meta.get_annotations()
    pprint(dict(list(ann.items())[:1]))
    print()

    # load activityner hierarchy
    hierarchy = load_hierarchy(dataset_name)
    print(f"---------- Hierarchy: ({len(hierarchy.data)})")
    pprint(dict(list(hierarchy.data.items())[:1]))
    print()

    # load synsets
    synset_to_cls = {v["nodeId"]: v for v in classes_data}

    def get_synset_info(synset_name):
        cls = synset_to_cls[synset_name]
        return cls

    # load boxes
    font_size = 1.2

    def get_image(t_key, t_scale=500, verbose=False):
        image_file = meta.get_image_file(t_key)
        image = Image.open(image_file).convert("RGB")
        image_scale = scaler.scale_image_bigger_side(image, t_scale)
        return image_scale

    def show_datapoint(t_key, show_text=False):
        image_scale = get_image(t_key)
        display(image_scale)
        t_class_idx = g_targets[t_key]
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
        "ann": ann,
        "get_synset_info": get_synset_info,
        "get_image": get_image,
        "show_datapoint": show_datapoint,
        "pr": pr,
        "scaler": scaler,
        "num2key": meta.get_num2key(),
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "hierarchy": hierarchy,
    }
    return namedtuple("ActivityNetData", output_dict.keys())(*output_dict.values())
