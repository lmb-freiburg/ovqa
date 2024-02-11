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


@define(slots=False)
class ActivityNetHierarchy(HierarchyInterface):
    """
    data:
    {
    "389": {
        "parentName": "Health-related self care",
        "nodeName": "Applying sunscreen",
        "nodeId": 389,
        "parentId": 269,
        "node_type": "leaf",
        "id": 199,
        "definition": "The act of applying a special cream that protects from the UV light from the sun.",
        "synonyms": ["Applying sunscreen", "Using sunprotection", "applying sunscream"]
    },
    "18": {
        "nodeId": 18,
        "parentName": "Eating and drinking Activities",
        "nodeName": "Food & Drink Prep., Presentation, & Clean-up",
        "parentId": 4,
        "children": [46, 31, 43],
        "node_type": "internal",
        "definition": "Engaging in tasks related to preparing, presenting, and cleaning up after food and beverages, often involving cooking, serving, and tidying.",
        "synonyms": ["Food & Drink Prep., Presentation, & Clean-up", "Food and beverage handling", "Meal preparation and serving"]

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
    freqs: Dict[str, float] = None
    classes_data: Dict[str, Any] = None

    @classmethod
    def load(cls, root: Path = get_ovqa_annotations_dir() / "activitynet"):
        t1 = timer()
        file = root / "simple_activitynet_hierarchy.json"
        data = load_json(file)

        # classes_data = load_json(root / f"generated/classes_data.json")
        classes_data = {}
        for wn_id in data.keys():
            # make sure all nodeIds are strings
            nodeId = data[wn_id]["nodeId"]
            if not isinstance(nodeId, str):
                nodeId = str(nodeId)
                data[wn_id]["nodeId"] = nodeId
            assert nodeId == wn_id, f"Keys and nodeIds should match (key {wn_id}, nodeId {nodeId})"

            # make sure all parentIds are strings
            parentId = data[wn_id]["parentId"]
            if parentId is not None and not isinstance(parentId, str):
                parentId = str(parentId)
                data[wn_id]["parentId"] = parentId

            # make all names lowercase
            data[wn_id]["nodeName"] = data[wn_id]["nodeName"].lower()
            data[wn_id]["synonyms"] = list(set([s.lower() for s in data[wn_id]["synonyms"]]))

            # merge all synonyms
            syns = {}
            for field in ["nodeName", "synonyms"]:
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
            data_dict["name"] = data_dict["nodeName"]
            parent_key = data_dict["parentId"]
            node_type = "leaf"
            if "children" in data_dict and len(data_dict["children"]) > 0:
                node_type = "internal"
            if parent_key is None:
                assert root_id is None
                root_id = wnid
                node_type = "root"
            else:
                G.add_edge(parent_key, wnid)
            G.add_node(wnid, label=data_dict["nodeName"], node_type=node_type)
            data_dict["node_type"] = node_type
            if node_type == "leaf":
                classes_data[int(data_dict["id"])] = data_dict
        assert root_id is not None
        G.root_id = root_id
        print(G)

        # add depth to the data
        def _walk(node_id, depth):
            data[node_id]["depth"] = depth
            for child_id in G.successors(node_id):
                _walk(child_id, depth + 1)

        _walk(root_id, 0)

        # load word frequencies
        # freqs = load_json(root / "class_hierarchy/word_freqs.json")
        freqs = {}

        # # load tiered imagenet
        # tiered_hierarchy = load_json(root / "class_hierarchy/v1/tiered_imagenet.json")
        classes_data = [classes_data[i] for i in range(len(classes_data))]
        td = timer() - t1
        logger.info(f"Loaded {len(data)} ActivityNet entries in {td:.2f} seconds")

        return cls(root, data, search, G, freqs, classes_data)  # , tiered_hierarchy)

    def get_all_class_keys(self) -> List[str]:
        return list(self.data.keys())

    def get_leaf_class_keys(self) -> List[str]:
        return [v["nodeId"] for v in self.classes_data]

    def get_class(self, class_key: str) -> Dict[str, Any]:
        dt = self.data[class_key]
        return {
            "class_key": class_key,
            "class_name": dt["name"],
            "parent_key": dt["parentId"],
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
    hier = ActivityNetHierarchy.load()
    for clsid, clsval in hier.data.items():
        if clsval["node_type"] == "leaf":
            continue
        print(f"{clsid} {clsval['name']}")
        for cclsid in hier.get_child_keys(clsid):
            cclsval = hier.data[cclsid]
            print(f"    {cclsid} {cclsval['name']}")

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    main()
