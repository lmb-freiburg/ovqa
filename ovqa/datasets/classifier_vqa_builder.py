from ovqa.common.lavis.registry import registry
from ovqa.datasets.activitynet_vqa_dataset import ActivityNetVQADataset
from ovqa.datasets.classifier_vqa_dataset import ClassifierVQADataset
from ovqa.datasets.coco_objects_vqa_dataset import COCOObjectsVQADataset, OVADAttributesVQADataset
from ovqa.datasets.lavis.base_dataset_builder import BaseDatasetBuilder


@registry.register_builder("imagenet1k")
class ImagenetVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = None
    eval_dataset_cls = ClassifierVQADataset

    DATASET_CONFIG_DICT = {
        "default": "NOT_DEFINED_YET",
        "eval": "ovqa/configs/datasets/imagenet1k.yaml",
    }


@registry.register_builder("coco")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = None
    eval_dataset_cls = COCOObjectsVQADataset

    DATASET_CONFIG_DICT = {
        "default": "NOT_DEFINED_YET",
        "eval": "ovqa/configs/datasets/coco.yaml",
    }


@registry.register_builder("ovad_attributes")
class OVADAttributesVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = None
    eval_dataset_cls = OVADAttributesVQADataset

    DATASET_CONFIG_DICT = {
        "default": "NOT_DEFINED_YET",
        "eval": "ovqa/configs/datasets/ovad_attributes.yaml",
    }


@registry.register_builder("activitynet")
class ActivityNetVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = None
    eval_dataset_cls = ActivityNetVQADataset

    DATASET_CONFIG_DICT = {
        "default": "NOT_DEFINED_YET",
        "eval": "ovqa/configs/datasets/activitynet.yaml",
    }
