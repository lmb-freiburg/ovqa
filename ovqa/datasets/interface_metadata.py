from typing import Optional, Dict, Any, List

from attr import define, field, fields_dict
from typedparser.objects import big_obj_to_short_str


def render_image_dummy(image_filename: str):
    print(f"render_image_dummy: {image_filename}")


@define(slots=False)
class ClsMetadataInterface:
    """
    Interface for metadata of a dataset.

    annotations should look like:
        { "key0": {
            "class_idx": 0,
            "image": "relative_path/to/image.jpg",  # relative to dataset dir
            "datapoint_num": 0,  # required for lavis
            # "class_name": "house",  # field name depends on label_field_name.
            # actually this should be in classes_data
        }, ...}
    classes_data should look like:
    [{"class_name": "house"}, ...]  # and optionally other information about the classes

    synonym_dict:
        {synonym: class_idx, ...} e.g. {"house": 0, "home": 0, ...}

    """

    annotations: Optional[Dict[str, Dict[str, Any]]] = field(
        default=None, repr=big_obj_to_short_str
    )
    classes_data: List[Dict[str, str]] = field(repr=big_obj_to_short_str, default=None)
    label_field_name: str = "class_name"
    bbox_data: Optional[Dict[str, Any]] = field(default=None, repr=big_obj_to_short_str)
    synonym_dict: Optional[Dict[str, int]] = field(default=None, repr=big_obj_to_short_str)
    antonym_dict: Optional[Dict[str, int]] = field(default=None, repr=big_obj_to_short_str)
    templates_name: str = "openai_imagenet_template"
    dataset_split: str = "unset"

    def get_class_list(self) -> List[str]:
        """
        Returns:
            List of class names
        """
        return [c[self.label_field_name] for c in self.classes_data]

    def get_targets(self) -> Dict[str, int]:
        """
        Returns:
            Dictionary {datapoint_key (str): class_idx (int)}
        """
        return {k: v["class_idx"] for k, v in self.annotations.items()}

    def get_annotations(self) -> Dict[str, Dict[str, Any]]:
        return self.annotations

    def get_image_file(self, root_id):
        raise NotImplementedError

    def get_num2key(self):
        """
        Get the mapping from datapoint number to datapoint key  e.g. {0: "val_00000001"}.
        """
        num2key = {int(v["datapoint_num"]): str(k) for k, v in self.annotations.items()}
        return num2key

    def new_copy(self, new_annotations):
        """Create a copy of the dataset with different metadata (e.g. to view a subset)"""
        new_self = {"annotations": new_annotations}
        for field_name in fields_dict(type(self)):
            if field_name in new_self:
                continue
            # new_self[field_name] = deepcopy(getattr(self, field_name))
            new_self[field_name] = getattr(self, field_name)  # assuming ref is fine here
        new_obj = type(self)(**new_self)
        return new_obj

    def show_datapoint(self, leaf_id: int, print_fn=print, image_fn=render_image_dummy):
        self.show_datapoint_visual(leaf_id, image_fn)
        self.show_datapoint_text(leaf_id, print_fn)

    def show_datapoint_visual(self, leaf_id: int, image_fn=render_image_dummy):
        if image_fn is None:
            return
        image_file = self.get_image_file(leaf_id)
        image_fn(image_file)

    def show_datapoint_text(self, leaf_id: int, print_fn=print):
        print_fn(self.get_datapoint_text(leaf_id))

    def get_datapoint_text(self, leaf_id: int) -> str:
        leaf_item = self.annotations[leaf_id]
        return f"Datapoint: {leaf_id} -> {leaf_item}"
