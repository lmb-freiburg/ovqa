"""
ImageNet 2012 val set

The class labels are given by https://github.com/LAION-AI/CLIP_benchmark

Other included class labels are from https://github.com/anishathalye/imagenet-simple-labels
(full synset, keras, or simple)

Class numbers are weird, there is a "clsidx" from 0-999 which maps to the labels found everywhere,
and a "clsnum" from 1-1000 in the original metadata, with a different order


Debugging class numbers help:
    tench is idx 0, old 449, n0144...
    seasnake is idx 65 old 490

datapoint val00000000001 is class idx 65
datapoint val_00000293 is class_idx 0,

"""
import os
import shutil
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from attr import define
from loguru import logger
from scipy.io import loadmat
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from ovqa.datasets.interface_metadata import ClsMetadataInterface
from ovqa.paths import get_ovqa_annotations_dir
from packg.iotools import yield_lines_from_file
from packg.iotools.jsonext import load_json, dump_json
from packg.paths import get_data_dir
from visiontext.bboxes import convert_bbox_abs_to_rel, convert_bbox_rel_to_abs

IMGNET_DEFAULT_LABEL = "clip_bench_label"


@define(slots=False)
class ImagenetClsMetadata(ClsMetadataInterface):
    @classmethod
    def load_split(
        cls,
        dataset_split: str,
        label_field_name: str = IMGNET_DEFAULT_LABEL,
        image_version: Optional[str] = None,
    ):
        available_splits = ["val"]
        assert (
            dataset_split in available_splits
        ), f"Split {dataset_split} not implemented, available: {available_splits}"
        annotations, classes_data = load_imagenet()
        classes = [c[label_field_name] for c in classes_data]

        for i, (ann_id, ann_item) in enumerate(annotations.items()):
            # preprocess classes
            clsidx = ann_item["class_idx"]
            class_name = classes[clsidx]
            ann_item["class_name"] = class_name

            # add some fields required
            ann_item["datapoint_num"] = i

            image_file = ann_item["image"]
            if image_version is not None:
                image_file = f"{image_version}/{image_file}"

            ann_item["image"] = image_file
            ann_item["image_file"] = image_file

        if dataset_split != "val":
            # filter given ids from data
            filter_ids = load_json(
                get_ovqa_annotations_dir() / f"imagenet1k/subsplits/ids_{dataset_split}.json"
            )
            new_ann = {k: annotations[k] for k in filter_ids}
            annotations = new_ann

        # load boxes
        try:
            bbox_file = (
                get_ovqa_annotations_dir() / f"imagenet1k/generated/bboxes_{dataset_split}.json"
            )
            bbox_data = load_json(bbox_file)
        except FileNotFoundError:
            logger.warning(f"Could not find bbox file for split {dataset_split}")
            bbox_data = None

        # load synonyms
        synonym_dict = load_json(get_ovqa_annotations_dir() / "imagenet1k/synonyms/base.json")

        # for now set leaf and root meta both with the same data, since it is a 1-1 relation here
        return cls(annotations, classes_data, label_field_name, bbox_data, synonym_dict)

    @staticmethod
    def get_dataset_dir():
        return get_data_dir() / "imagenet1k"

    def get_image_file(self, root_id):
        image_file = self.get_dataset_dir() / self.annotations[root_id]["image_file"]
        return image_file

    def load_image_with_box(self, leaf_id):
        image_file = self.get_image_file(leaf_id)
        image = Image.open(image_file)
        real_image_w, real_image_h = image.size
        rx, ry, rw, rh = self.get_biggest_box_rel(leaf_id)
        fx, fy, fw, fh = convert_bbox_rel_to_abs(rx, ry, rw, rh, real_image_w, real_image_h)
        x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh

        image_np = np.asarray(image)
        avg_color = np.mean(image_np)
        if avg_color > 128:
            box_color = (0, 0, 0)
        else:
            box_color = (255, 255, 255)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)

        image_tensor = torch.permute(torch.from_numpy(image_np), (2, 0, 1))
        image_tensor_boxes = draw_bounding_boxes(
            image_tensor,
            torch.tensor([[x1, y1, x2, y2]]),
            [""],
            [box_color],
            font_size=20,
            font="/misc/lmbssd/gings/cv_shared_data/font/DejaVuSans.ttf",
            width=3,
        )
        image = Image.fromarray(torch.permute(image_tensor_boxes, (1, 2, 0)).numpy())
        return image

    def get_biggest_box_rel(self, leaf_id: int) -> Tuple[float, float, float, float]:
        bbox_item = self.bbox_data[leaf_id]
        image_h, image_w = bbox_item["image_h"], bbox_item["image_w"]
        objects = bbox_item["objects"]
        biggest_area, coords = 0, None
        for obj in objects:
            bbox = obj["bbox"]
            bx, by, bw, bh = bbox
            area = bw * bh
            if area < biggest_area:
                continue
            biggest_area = area
            rx, ry, rw, rh = convert_bbox_abs_to_rel(bx, by, bw, bh, image_w, image_h)
        return rx, ry, rw, rh  # noqa

    def get_datapoint_for_print(self, leaf_id: int) -> Dict:
        leaf_item = self.annotations[leaf_id]
        class_info = self.classes_data[leaf_item["class_idx"]]
        coll = {}
        for field_name, field_title in [
            ("simple_label", "Simple"),
            ("synset_label", "Synset"),
            ("keras_label", "Keras"),
            ("synset", "Synset ID"),
            ("orig_descr", "Description"),
        ]:
            coll[field_title] = class_info[field_name]
        return coll

    def get_datapoint_text(self, leaf_id: int):
        leaf_item = self.annotations[leaf_id]
        fields = self.get_datapoint_for_print(leaf_id)
        coll = []
        for field_title, field_value in fields.items():
            coll.append(f"{field_title}: {field_value}")
        out_str = f"{leaf_item['class_name']}" + f" | ".join(coll)
        return out_str

    def create_meta_bin(self):
        imagenet_dir = self.get_dataset_dir()
        create_meta_bin(imagenet_dir)


def load_imagenet(overwrite=False):
    """
    Returns:
        ann: {"val_00000001": {'class_num': 489, 'image': 'val/ILSVRC2012_val_00000001.JPEG'}, ...}
        classes_data: list of dicts with class info, e.g.
            {'clsidx': 0, 'clsnum': 449, 'synset': 'n01440764',
             'synset_label': 'tench, Tinca tinca', 'keras_label': 'tench', 'simple_label': 'tench',
             'orig_label': 'tench, Tinca tinca',
             'orig_descr': 'freshwater dace-like game fish of ...'}
    """
    split = "val"
    path = get_data_dir() / "imagenet1k"
    anno_path = get_ovqa_annotations_dir() / "imagenet1k"
    classes_file = anno_path / "generated/classes_data.json"
    if not classes_file.is_file() or overwrite:
        classes_data = load_imagenet_classes(path, anno_path)
        dump_json(classes_data, classes_file, create_parent=True, indent=2)
    classes_data = load_json(classes_file)
    clsnum_to_clsidx = {a["old_class_num"]: i for i, a in enumerate(classes_data)}

    # sort images if necessary
    sorted_target = path / "val"
    if not sorted_target.is_dir():
        unsorted_source = path / "unsorted_val"
        assert unsorted_source.is_dir(), (
            f"Directory {unsorted_source} does not exist. Create with:\n"
            f"mkdir unsorted_val && tar -xf ILSVRC2012_img_val.tar -C unsorted_val"
        )
        filemap_file = anno_path / "filemap" / "map_id_to_imagefile-val.json"
        filemap = load_json(filemap_file)

        unsorted_source = unsorted_source
        sorted_target = sorted_target
        for file in tqdm(sorted(os.listdir(unsorted_source)), desc="Sorting images (move)"):
            # file val_00000009.jpg
            image_id = file.split(".")[0]
            if image_id.startswith("ILSVRC2012_"):
                image_id = image_id[len("ILSVRC2012_") :]
            full_file = unsorted_source / file
            target_file = sorted_target / filemap[image_id]
            os.makedirs(target_file.parent, exist_ok=True)
            shutil.move(full_file, target_file)

    ann_file = anno_path / f"generated/{split}.json"
    if not ann_file.is_file() or overwrite:
        # labels also start with 1 i.e. [1, 2, ..., 1000]
        label_file = path / "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
        assert label_file.is_file(), f"File {label_file} does not exist"
        labels = [int(a) for a in yield_lines_from_file(label_file)]
        images_unsorted = [
            a.relative_to(path).as_posix() for a in sorted(sorted_target.glob("**/*.JPEG"))
        ]
        assert len(labels) == len(
            images_unsorted
        ), f"Got {len(labels)} labels but {len(images_unsorted)} images"
        image_ids = [str(a.split("_", maxsplit=1)[1].split(".")[0]) for a in images_unsorted]
        # now images must be sorted by image_ids to match the labels
        id_to_image_sorted = {
            a: b for a, b in sorted(zip(image_ids, images_unsorted), key=lambda x: x[0])
        }
        ann = {}
        for i, (image_id, image) in enumerate(id_to_image_sorted.items()):
            clsnum = labels[i]
            clsidx = clsnum_to_clsidx[clsnum]
            ann[image_id] = {
                "class_idx": clsidx,
                "image": f"{id_to_image_sorted[image_id]}",
            }
            pass
        dump_json(ann, ann_file, indent=2)
    ann = load_json(ann_file)

    return ann, classes_data


def load_imagenet_classes(path, anno_path):
    # get the class number to original data mapping
    # this includes some other synsets (1860 classes in total)
    meta = loadmat((path / "ILSVRC2012_devkit_t12/data/meta.mat").as_posix())
    synsets = meta["synsets"]
    clsnum_to_synset_all, clsnum_to_label, clsnum_to_descr = {}, {}, {}
    for s in synsets:
        idx = int(s[0][0][0][0])  # class number starting with 1
        wnid = s[0][1][0]  # wordnet synset e.g. n02012849
        clsnum_to_synset_all[idx] = wnid
        clsnum_to_label[idx] = s[0][2][0]
        clsnum_to_descr[idx] = s[0][3][0]
    assert len(clsnum_to_synset_all) == 1860  # 1000

    # load the clsidx -> (synset, keras_label) mapping
    clsidx_to_synset_and_keras: Dict[str, Tuple[str, str]] = load_json(
        anno_path / "external/imagenet_class_index.json"
    )
    clsidx_to_synset, clsidx_to_keras = {}, {}

    for clsidx_str, (synset, keras_label) in clsidx_to_synset_and_keras.items():
        clsidx = int(clsidx_str)
        clsidx_to_synset[clsidx] = synset
        clsidx_to_keras[clsidx] = keras_label

    # build clsnum_to_clsidx mapping (only for 1000 actual classes)
    synset_to_clsidx = {v: k for k, v in clsidx_to_synset.items()}
    clsnum_to_clsidx = {
        k: synset_to_clsidx[v] for k, v in clsnum_to_synset_all.items() if v in synset_to_clsidx
    }

    clsidx_to_content = {}
    for clsnum, clsidx in clsnum_to_clsidx.items():
        clsidx_to_content[clsidx] = {
            "old_class_num": clsnum,
            "class_idx": clsidx,
            "synset": clsidx_to_synset[clsidx],
            "orig_descr": clsnum_to_descr[clsnum],
        }
    assert len(clsidx_to_content) == 1000
    all_classes = [clsidx_to_content[clsidx] for clsidx in range(1000)]

    # load clip benchmark labels list
    clipbenchlabels = load_json(anno_path / "external" / "clip_benchmark_classes_fixed.json")
    assert len(clipbenchlabels) == 1000

    for clsidx, content in clsidx_to_content.items():
        content["clip_bench_label"] = clipbenchlabels[clsidx]
    return all_classes


def create_meta_bin(imagenet_dir):
    meta_bin_file = imagenet_dir / "meta.bin"
    if not meta_bin_file.is_file():
        logger.info(f"Parse devkit archive to create {meta_bin_file}")
        from torchvision.datasets.imagenet import parse_devkit_archive

        parse_devkit_archive(imagenet_dir.as_posix())


def main():
    ImagenetClsMetadata.load_split("val")


if __name__ == "__main__":
    main()
