import re
from collections import defaultdict
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, Any, List

import networkx as nx
from attr import define
from loguru import logger

from packg.iotools.jsonext import load_json
from packg.paths import get_data_dir
from ovqa.paths import get_ovqa_annotations_dir


@define(slots=False)
class HierarchyInterface:
    """
    Note:
        class_key for imagenet is the wordnet_id
    """

    def get_all_class_keys(self) -> List[str]:
        """Get all class keys (both parent and leaves)"""

    def get_leaf_class_keys(self) -> List[str]:
        """Get keys for the leafs (classes used in classification)
        in the same order as the classes in the dataset"""

    def get_class(self, class_key: str) -> Dict[str, Any]:
        """Get information for a single class node

        {
            "class_name": "tench",
            "parent_id": "n01439121",
            "synonyms": ["tench", "Tinca tinca"],  # including original name
            "node_type": "leaf",  # or "internal"
        }
        """

    def get_parent_keys(self, class_key: str) -> List[str]:
        """Get all parent keys (not including the input key)"""

    def get_child_keys(self, class_key: str) -> List[str]:
        """Get all child keys recursively

        Only needed for not implemented consider_neighbors > 0
        """
