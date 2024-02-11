from collections import namedtuple
from pprint import pprint

from IPython.display import Image, display
from PIL import Image

from visiontext.htmltools import NotebookHTMLPrinter
from packg.iotools.jsonext import load_json
from ovqa.paths import get_ovqa_annotations_dir
from ovqa.datasets.interface_metadata import ClsMetadataInterface
from ovqa.datasets.meta_loading import meta_loader
from visiontext.bboxes import (
    convert_bbox_abs_to_rel,
    convert_bbox_rel_to_abs,
    get_bbox_bounds,
    draw_bounding_boxes_pil,
)
from visiontext.images import PILImageScaler


def load_imagenet_for_notebook(dataset_name="imagenet1k", dataset_split="val"):
    pr = NotebookHTMLPrinter()
    scaler = PILImageScaler(return_pillow=True)

    meta: ClsMetadataInterface = meta_loader.load_metadata(dataset_name, dataset_split)
    classes_data = meta.classes_data
    print(classes_data[0])

    # dataset size
    num_datapoints = len(meta.annotations)
    print(f"Size of dataset: {num_datapoints}")

    # class names
    print(f"---------- Class names:")
    class_names = meta.get_class_list()
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

    # load synsets
    synset_to_cls = {v["synset"]: v for v in classes_data}

    def get_synset_info(synset_name):
        cls = synset_to_cls[synset_name]
        return cls

    # load boxes
    bbox_file = get_ovqa_annotations_dir() / "imagenet1k/generated/bboxes_val.json"
    bbox_data = load_json(bbox_file)
    font_size = 1.2

    def get_image(
        t_key,
        t_scale=500,
        verbose=False,
        all_boxes=False,
        show_boxes=True,
        which_boxes=(
            False,
            True,
        ),
    ):
        # assert crop is None, f"No crops defined for this dataset {dataset_name}"
        # item = meta.annotations[t_key]
        # t_class_idx = targets[t_key]
        image_file = meta.get_image_file(t_key)
        image = Image.open(image_file).convert("RGB")

        for create_squares in False, True:
            if not show_boxes:
                break
            real_image_w, real_image_h = image.size
            # print(f"real_image_h={real_image_h}, real_image_w={real_image_w}")

            bbox_item = bbox_data[t_key]
            image_h, image_w = bbox_item["image_h"], bbox_item["image_w"]
            # print(f"image_h={image_h}, image_w={image_w}")
            objects = bbox_item["objects"]
            if verbose:
                print(f"Squares: {create_squares} image hxw {image_h}x{image_w} ")

            box_color = (0, 255, 0) if create_squares else (255, 0, 0)

            min_box = 64
            box_coords, box_labels, box_colors = [], [], []
            biggest_area = 0
            for iobj, obj in enumerate(objects):
                synset_name = obj["synset"]
                bbox = obj["bbox"]
                class_info = get_synset_info(synset_name)
                bx, by, bw, bh = bbox
                rx, ry, rw, rh = convert_bbox_abs_to_rel(bx, by, bw, bh, image_w, image_h)
                fx, fy, fw, fh = convert_bbox_rel_to_abs(rx, ry, rw, rh, real_image_w, real_image_h)
                x1, y1, x2, y2 = get_bbox_bounds(
                    fx,
                    fy,
                    fw,
                    fh,
                    real_image_w,
                    real_image_h,
                    min_w=min_box,
                    min_h=min_box,
                    create_squares=create_squares,
                )
                box = [x1, y1, x2 - x1, y2 - y1]
                box_coords.append(box)
                # box_labels.append(f"{synset_name} {class_info['clip_bench_label']}")
                box_labels.append(f"{class_info['clip_bench_label']}")
                box_colors.append(box_color)
                box_area = box[2] * box[3]
                if box_area > biggest_area and not create_squares:
                    # search biggest box only in rectangle mode first, then keep it
                    biggest_area = box_area
                    biggest_box_id = iobj
                elif verbose:
                    print(f"Skip box area {box_area} < {biggest_area} with box {box}")

            if all_boxes:
                box_labels = "numbers"
            else:
                box_coords = [box_coords[biggest_box_id]]
                # box_labels = [box_labels[biggest_box_id]]
                box_colors = [box_colors[biggest_box_id]]
                box_labels = None
            if create_squares in which_boxes:
                image = draw_bounding_boxes_pil(
                    image,
                    box_coords,
                    box_labels,
                    box_colors,
                )
        image_scale = scaler.scale_image_bigger_side(image, t_scale)
        return image_scale

    def show_datapoint(
        t_key,
        show_text=False,
        which_boxes=(
            False,
            True,
        ),
        t_scale=500,
    ):
        image_scale = get_image(t_key, which_boxes=which_boxes, t_scale=t_scale)
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
        "bbox_data": bbox_data,
        "num2key": meta.get_num2key(),
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
    }
    return namedtuple("ImagenetData", output_dict.keys())(*output_dict.values())
