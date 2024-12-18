"""
This script converts the imagenet bounding box format to json
and crops the imagenet images given the biggest box.

ImageNet format:
    ImageNet/bboxes/val/ILSVRC2012_val_00000001.xml
    ...

A single XML contains the following data:
    {'filename': 'ILSVRC2012_val_00050000',
     'folder': 'val',
     'objects': [{'bndbox': {'xmax': '446',
                             'xmin': '237',
                             'ymax': '374',
                             'ymin': '109'},
                  'difficult': '0',
                  'name': 'n02437616',
                  'pose': 'Unspecified',
                  'truncated': '0'},
                 {'bndbox': {'xmax': '474',
                             'xmin': '175',
                             'ymax': '342',
                             'ymin': '89'},
                  'difficult': '0',
                  'name': 'n02437616',
                  'pose': 'Unspecified',
                  'truncated': '0'}],
     'segmented': '0',
     'size': {'depth': '3', 'height': '375', 'width': '500'},
     'source': {'database': 'ILSVRC_2012'}}

The final output file will contain all data as follows, with coco bbox format (x, y, w, h):
    {
        "val_00050000": {"image_h": 375, "image_w": 500, "objects": [
            {"synset": "n02437616", "bbox": [237, 109, 209, 265]},
            {"synset": "n02437616", "bbox": [175, 89, 299, 253]}]},
    }

Boxes downloaded from:
    https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
    Validation bounding box annotations (all tasks) . 2.2MB. MD5: f4cd18b5ea29fe6bbea62ec9c20d80f0
    https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz
"""

import os
from PIL import Image
from attr import define
from lxml import etree
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ovqa.datasets.interface_metadata import ClsMetadataInterface
from ovqa.datasets.meta_loading import meta_loader
from ovqa.paths import get_ovqa_annotations_dir
from packg.iotools.jsonext import dump_json, load_json
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from ovqa.paths import get_data_dir
from typedparser import VerboseQuietArgs, TypedParser, add_argument
from visiontext.bboxes import convert_bbox_abs_to_rel, convert_bbox_rel_to_abs, get_bbox_bounds


@define(slots=False)
class Args(VerboseQuietArgs):
    data_dir: Optional[Path] = add_argument(
        shortcut="-d", type=str, help="Source base dir", default=None
    )
    crop_min_size: int = add_argument(shortcut="-c", type=int, help="Crop min size", default=64)
    crop_rect: bool = add_argument(
        shortcut="-r", action="store_true", help="Crop rectangles instead of squares"
    )


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)

    dataset_name = "imagenet1k"
    data_dir = get_data_dir() / dataset_name if args.data_dir is None else args.data_dir
    anno_dir = get_ovqa_annotations_dir() / dataset_name

    split = "val"
    output_file = anno_dir / "generated" / f"bboxes_{split}.json"
    if not output_file.is_file():
        data = read_2012_boxes(data_dir)
        os.makedirs(output_file.parent, exist_ok=True)
        dump_json(data, output_file)
    else:
        print(f"File {output_file} already exists, skipping saving.")
        data = load_json(output_file)

    crop_images(
        data,
        data_dir,
        crop_min_size=args.crop_min_size,
        crop_rect=args.crop_rect,
        split=split,
    )


def read_2012_boxes(data_dir):
    # read the 2012 official boxes
    bbox_dir = data_dir / "bboxes/val"  # ILSVRC2012_val_00036658.xml
    files = sorted(os.listdir(bbox_dir))
    data_out = {}
    for file in tqdm(files, desc="Reading box data"):
        assert file.startswith("ILSVRC2012_val_")
        key = file[len("ILSVRC2012_") : -len(".xml")]
        full_file = bbox_dir / file

        tree = etree.parse(full_file)
        root = tree.getroot()

        # print(etree.tostring(root, pretty_print=True, encoding="unicode"))
        data = tree_to_dict(root)
        # pprint(data)
        data_out[key] = read_box_data(data)
    return data_out


def crop_images(data_out, data_dir, crop_min_size: int = 64, crop_rect: bool = False, split="val"):
    # crop images
    meta: ClsMetadataInterface = meta_loader.load_metadata("imagenet1k", split)
    classes_data = meta.classes_data
    synset_to_cls = {v["synset"]: v for v in classes_data}

    mode = "rect" if crop_rect else "square"
    crop_name = f"{mode}"

    # print(f"synset_name={synset_name}, cls={cls}")
    output_dir = data_dir / f"{crop_name}"
    if output_dir.is_dir():
        raise FileExistsError(f"Output dir {output_dir} already exists")
    pbar = tqdm(total=len(meta.annotations), desc=f"Cropping images to {output_dir}")
    stats = {"w": [], "h": [], "area": []}
    for i, (key, meta_item) in enumerate(meta.annotations.items()):
        image_class_idx = meta_item["class_idx"]
        image_file = meta.get_image_file(key)
        image = Image.open(image_file)
        real_image_w, real_image_h = image.size

        bbox_item = data_out[key]
        image_h, image_w = bbox_item["image_h"], bbox_item["image_w"]
        objects = bbox_item["objects"]
        biggest_area, coords = 0, None
        for box_obj in objects:
            synset_name = box_obj["synset"]
            cls = synset_to_cls[synset_name]
            class_idx = cls["class_idx"]
            assert (
                class_idx == image_class_idx
            ), f"Class mismatch: Box {class_idx} != Image {image_class_idx}"
            bbox = box_obj["bbox"]
            bx, by, bw, bh = bbox
            area = bw * bh
            if area < biggest_area:
                continue
            biggest_area = area
            rx, ry, rw, rh = convert_bbox_abs_to_rel(bx, by, bw, bh, image_w, image_h)
            fx, fy, fw, fh = convert_bbox_rel_to_abs(rx, ry, rw, rh, real_image_w, real_image_h)

            x1, y1, x2, y2 = get_bbox_bounds(
                fx,
                fy,
                fw,
                fh,
                real_image_w,
                real_image_h,
                min_w=crop_min_size,
                min_h=crop_min_size,
                create_squares=not crop_rect,
            )
            coords = x1, y1, x2, y2
            stats["w"].append(fw)
            stats["h"].append(fh)
            stats["area"].append(fw * fh)
        assert coords is not None, f"Could not find coords for {key}"
        x1, y1, x2, y2 = coords
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_file = output_dir / meta_item["image_file"]
        os.makedirs(cropped_file.parent, exist_ok=True)
        cropped_image.save(cropped_file)
        pbar.update(1)
    pbar.close()

    # for key, data in stats.items():
    #     print(f"********** {key} **********")
    #     print(pd.Series(data).describe())


def read_box_data(data):
    image_h, image_w = int(data["size"]["height"]), int(data["size"]["width"])

    assert int(data["segmented"]) == 0, f"Expected segmented=0, got {data['segmented']}"
    objects = data["object"]
    object_out = []
    for object_data in objects:
        box_data = object_data["bndbox"]
        xmin, xmax, ymin, ymax = (
            int(box_data["xmin"]),
            int(box_data["xmax"]),
            int(box_data["ymin"]),
            int(box_data["ymax"]),
        )
        assert (
            int(object_data["difficult"]) == 0
        ), f"Expected difficult=0, got {object_data['difficult']}"
        synset_name = object_data["name"]
        pose = object_data["pose"]
        assert pose == "Unspecified", f"Expected pose=Unspecified, got {pose}"
        truncated = int(object_data["truncated"])
        assert truncated == 0, f"Expected truncated=0, got {truncated}"

        # convert to coco format (x, y, w, h)
        w = xmax - xmin
        h = ymax - ymin

        object_out.append(
            {
                "synset": synset_name,
                "bbox": [xmin, ymin, w, h],
            }
        )
    return {
        "image_h": image_h,
        "image_w": image_w,
        "objects": object_out,
    }


# in XML we do not know apriori whether an element occurs once or multiple times
# define here that all fields are singular, except the object field
_multiple_values = {0: ["object"]}


def tree_to_dict(root, depth=0):
    out = {}
    multi_fields = _multiple_values.get(depth, [])
    for field in multi_fields:
        out[f"{field}"] = []

    for child in root:
        child: etree.ElementBase
        assert len(child.attrib) == 0, f"Expected no attributes, got {child.attrib}"
        key = child.tag
        assert child.prefix is None, f"Expected no prefix, got {child.prefix}"
        assert child.tail.strip() == "", f"Expected no tail, got {child.tail}"
        text = child.text.strip()  # should have value only if there is no children
        # base is the path to the file, nsmap is some empty namespace mapping,
        # sourceline is the line in the file

        if len(child) == 0:
            assert text != "", f"Expected text, got no text"
            value = text
        else:
            assert text == "", f"Expected no text, got {text}"
            value = tree_to_dict(child, depth=depth + 1)

        if key in multi_fields:
            out[key].append(value)
        else:
            assert key not in out, f"Expected no duplicate keys, got {key}"
            out[key] = value

    return out


if __name__ == "__main__":
    main()
