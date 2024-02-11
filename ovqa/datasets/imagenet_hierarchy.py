"""
TBD"""

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

from ovqa.datasets.interface_hierarchy import HierarchyInterface


def convert_wn_id(wn_id: str):
    # tiered imagenet uses leading zeros while our existing hierarchy does not
    # fix this by using the 8 leading zeros
    if wn_id[0] != "n":
        raise ValueError(f"Expected wordnet id to start with 'n', got {wn_id}")
    return f"n{int(wn_id[1:]):08d}"


def format_frequency(freq: float) -> str:
    return f"{freq:.1e}".replace("e+0", "e+").replace("e-0", "e-")


def load_hierarchy(dataset_name, **kwargs):
    if dataset_name.startswith("imagenet"):
        return ImagenetHierarchy.load(**kwargs)
    if dataset_name.startswith("activitynet"):
        from ovqa.datasets.activitynet_hierarchy import ActivityNetHierarchy

        return ActivityNetHierarchy.load(**kwargs)
    raise ValueError(f"Unknown dataset {dataset_name}")


@define(slots=False)
class ImagenetHierarchy(HierarchyInterface):
    """
    data:
    {
      "n01440764": {
        "name": "tench",
        "id": "n01440764",
        "parent": "cyprinid",
        "parent_id": "n01439121",
        "synset": "tench.n.01",
        "lemmas": ["tench", "Tinca tinca"],
        "definition": "freshwater dace-like game fish of Europe and western Asia noted for ability to survive outside water",
        "labels": "tench",

        # these attributes are also added
        "node_type"
        "depth"
      },
    ...
    }
    """

    root: Path = None
    data: Dict[str, Any] = {}
    search_index: Dict[str, List[str]] = {}
    nx_graph: nx.DiGraph = None
    classes_data: Dict[str, Any] = None

    def get_all_class_keys(self) -> List[str]:
        return list(self.data.keys())

    def get_leaf_class_keys(self) -> List[str]:
        return [v["synset"] for v in self.classes_data]

    def get_class(self, class_key: str) -> Dict[str, Any]:
        dt = self.data[class_key]
        return {
            "class_key": class_key,
            "class_name": dt["name"],
            "parent_key": dt["parent_id"],
            "synonyms": self.get_synonyms(class_key),
            "node_type": dt["node_type"],
            "depth": dt["depth"],
        }

    def get_parent_keys(self, wn_id: str):
        def _climb(node_id: str):
            # the edges point from parent to child
            parents = list(self.nx_graph.predecessors(node_id))
            assert len(parents) <= 1  # always 1 except for root
            if len(parents) == 1:
                parent_key = parents[0]
                return [parent_key] + _climb(parent_key)
            else:
                return []

        return _climb(wn_id)

    def get_child_keys(self, wn_id: str):
        def _descend(node_id: str):
            # the edges point from parent to child
            children = list(self.nx_graph.successors(node_id))
            sub_children = []
            for child in children:
                sub_children += _descend(child)
            return children + sub_children

        return _descend(wn_id)

    @classmethod
    def load(cls, root: Path = get_ovqa_annotations_dir() / "imagenet1k"):
        t1 = timer()
        file = root / "class_hierarchy/simple_imagenet_hierarchy_with_labels.json"
        data = load_json(file)
        classes_data = load_json(root / f"generated/classes_data.json")
        # merge all synonyms (except conceptnet)
        for wn_id in data.keys():
            syns = {}
            for field in ["labels", "name", "lemmas"]:  # , "conceptnet_synonyms"]:
                if field not in data[wn_id]:
                    continue
                syns_field = data[wn_id][field]
                if isinstance(syns_field, str):
                    syns_field = [syns_field]
                for syn in syns_field:
                    if syn not in syns:
                        syns[syn] = field
            data[wn_id]["synonyms"] = syns

        # build search index for synonyms (dict from synonym to list of wn_ids)
        search = defaultdict(list)
        for wn_id in data.keys():
            syns = data[wn_id]["synonyms"]
            for syn in syns.keys():
                search[syn].append(wn_id)

        # build the graph
        G = nx.DiGraph()
        root_id = None
        for i, (wnid, data_dict) in enumerate(data.items()):
            parent_key = data_dict["parent_id"]
            node_type = "leaf"
            if "children" in data_dict and len(data_dict["children"]) > 0:
                node_type = "internal"
            if parent_key is None:
                assert root_id is None
                root_id = wnid
                node_type = "root"
            else:
                G.add_edge(parent_key, wnid)
            G.add_node(wnid, label=data_dict["name"], node_type=node_type)
            data_dict["node_type"] = node_type
        assert root_id is not None
        G.root_id = root_id
        print(G)

        # add depth to the data
        def _walk(node_id, depth):
            data[node_id]["depth"] = depth
            for child_id in G.successors(node_id):
                _walk(child_id, depth + 1)

        _walk(root_id, 0)

        td = timer() - t1
        logger.info(f"Loaded {len(data)} ImageNet entries in {td:.2f} seconds")
        return cls(root, data, search, G, classes_data)  # , tiered_hierarchy)

    def search(self, term):
        term_regex = re.compile(term)
        # build search results (dict from wn_id to list of terms that matched)
        found_ids = defaultdict(set)
        for key in self.search_index.keys():
            if term_regex.fullmatch(key):
                wn_ids = self.search_index[key]
                for wn_id in wn_ids:
                    if term not in found_ids[wn_id]:
                        found_ids[wn_id].add(key)
        logger.info(f"Found {len(found_ids)} wordnet ids for {term}")
        return found_ids

    def get_synonyms(self, wn_id: str, exclude_name=False):
        synonym_dict = self.data[wn_id]["synonyms"]
        synonyms = list(synonym_dict.keys())
        # if not clean:
        # else:
        #     # possible sources right now are ["labels", "name", "lemmas", "conceptnet_synonyms"]
        #     synonyms = [
        #         k for k, v in synonym_dict.items() if v in ["labels", "name", "lemmas"]
        #     ]
        if exclude_name:
            synonyms = [s for s in synonyms if s != self.data[wn_id]["name"]]
        return synonyms


def main():
    hier = ImagenetHierarchy.load()
    for clsid, clsval in hier.data.items():
        if clsval["node_type"] == "leaf":
            continue
        print(f"{clsid} {clsval['name']}")
        for cclsid in hier.get_child_keys(clsid):
            cclsval = hier.data[cclsid]
            print(f"    {cclsid} {cclsval['name']}")


if __name__ == "__main__":
    main()
