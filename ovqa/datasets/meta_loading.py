from ovqa.datasets.activitynet import ActivityNet
from ovqa.datasets.imagenet import ImagenetClsMetadata
from ovqa.datasets.interface_metadata import ClsMetadataInterface
from ovqa.datasets.ovad2000 import OVAD2000
from packg import Const
from packg.paths import get_data_dir


class DatasetConst(Const):
    IMAGENET1K = "imagenet1k"
    OVAD_ATTRIBUTES = "ovad_attributes"
    # OVAD_OBJECTS = "ovad_objects"
    ACTIVITYNET = "activitynet"
    COCO = "coco"


class LazyMetaLoader:
    """
    This class loads and stores metadata for datasets.
    """
    def __init__(self):
        self.datasets = {}
        self.data_path = get_data_dir()

    def load_metadata(self, dataset_name, dataset_split, **kwargs) -> ClsMetadataInterface:
        store_key = f"{dataset_name}~{dataset_split}"
        if store_key not in self.datasets:
            if dataset_name.startswith(DatasetConst.IMAGENET1K):
                image_version = None
                if dataset_name != DatasetConst.IMAGENET1K:
                    image_version = dataset_name[len(DatasetConst.IMAGENET1K) + 1 :]
                metadata: ClsMetadataInterface = ImagenetClsMetadata.load_split(
                    dataset_split, image_version=image_version
                )
            # elif dataset_name == DatasetConst.OVAD_OBJECTS:
            #     class_obj_att = "object"
            #     metadata: ClsMetadataInterface = OVAD2000.load_split(
            #         dataset_split, class_obj_att=class_obj_att, **kwargs
            #     )
            elif dataset_name == DatasetConst.OVAD_ATTRIBUTES:
                class_obj_att = "attribute"
                metadata: ClsMetadataInterface = OVAD2000.load_split(
                    dataset_split, class_obj_att=class_obj_att, **kwargs
                )
            elif dataset_name == DatasetConst.ACTIVITYNET:
                metadata: ClsMetadataInterface = ActivityNet.load_split(dataset_split)
            elif dataset_name == DatasetConst.COCO:
                class_obj_att = "object"
                metadata: ClsMetadataInterface = OVAD2000.load_split(
                    dataset_split,
                    class_obj_att=class_obj_att,
                    json_file="coco/annotations/instances_val2017.json",
                    **kwargs,
                )
            else:
                raise NotImplementedError(
                    f"Dataset {dataset_name} not implemented, options: "
                    f"{list(DatasetConst.values())}"
                )
            self.datasets[store_key] = metadata
        return self.datasets[store_key]


meta_loader = LazyMetaLoader()
