import logging
import torch
from attr import define
from omegaconf import DictConfig, OmegaConf
from sentence_transformers.util import cos_sim
from tqdm import tqdm
from typing import Dict, Union, List, Any, Optional

from ovqa.datasets.imagenet_hierarchy import HierarchyInterface, ImagenetHierarchy
from ovqa.datasets.interface_metadata import ClsMetadataInterface
from ovqa.metrics.clip_match import compute_class_embeddings_given_embedder, EmbeddingAccuracy
from ovqa.metrics.clip_match_synonyms import EmbeddingSynonymAccuracy
from ovqa.metrics.syn_logits import reduce_synonym_logits_over_classes
from ovqa.paths import get_ovqa_annotations_dir
from ovqa.textutils.cut_sentence import cut_too_long_text
from ovqa.textutils.embeddings import EmbeddingsPackageConst, SentenceEmbedderInterface
from packg.iotools import load_json
from typedparser import attrs_from_dict


@define(slots=False)
class FollowupConfig:
    model_name: str = "EVA01-g-14/laion400m_s11b_b41k"  # "ViT-L-14/openai"
    package_name: str = EmbeddingsPackageConst.OPEN_CLIP

    use_synonyms_leaves: bool = False
    templates_name_leaves: str = "openai_imagenet_template"
    arg_max_or_average_syn_leaves: str = "arg_max_syn"

    use_synonyms_parents: bool = True
    templates_name_parents: str = "openai_imagenet_template"
    arg_max_or_average_syn_parents: str = "arg_max_syn"

    threshold: float = 0.53  # 0.53 seems ok for imagenet parents
    verbose: bool = True

    followup_prompt: str = "Question: What type of {} is this? Short answer:"
    default_followup_object: str = "object"
    cut_long_text: bool = True
    exclude_parents: Optional[list[str]] = None

    canonical_things: List[str] = []
    canonical_mode_leaves: str = None
    canonical_mode_parents: str = None

    @classmethod
    def from_config(
        cls, followup_cfg: Union[Dict[str, Any], DictConfig, "FollowupConfig"]
    ) -> "FollowupConfig":
        if isinstance(followup_cfg, cls):
            return followup_cfg
        if isinstance(followup_cfg, DictConfig):
            # resolve omega config to dictionary
            followup_cfg = OmegaConf.to_container(followup_cfg, resolve=True)
        # convert dictionary to this class (set strict=False to disable typechecking)
        return attrs_from_dict(cls, followup_cfg, strict=False)


@define(slots=False)
class Followup:
    cfg: Union[Dict[str, Any], DictConfig, "FollowupConfig"]
    hierarchy: HierarchyInterface
    leaf_class_names: List[str]
    leaf_synonym_dict: Optional[Dict[str, int]]
    targets: Dict[str, int]  # datapoint key to class number

    @classmethod
    def from_metadata(cls, cfg, hierarchy: HierarchyInterface, meta: ClsMetadataInterface):
        leaf_class_names = meta.get_class_list()
        leaf_synonym_dict = meta.synonym_dict
        targets = meta.get_targets()
        return cls(cfg, hierarchy, leaf_class_names, leaf_synonym_dict, targets)

    def __attrs_post_init__(self):
        self.cfg = FollowupConfig.from_config(self.cfg)

        # build leaf info from metadata
        self.leaf_class_name2num = {k: i for i, k in enumerate(self.leaf_class_names)}
        self.leaf_class_key2num = {k: i for i, k in enumerate(self.hierarchy.get_leaf_class_keys())}

        # build parent info from the hierarchy
        self.parent_class_names: List[str] = []
        self.parent_syns: List[List[str]] = []
        self.parent_class_key2num = {}
        for class_key in self.hierarchy.get_all_class_keys():
            v = self.hierarchy.get_class(class_key)
            class_name = v["class_name"]
            # remove leaf and root nodes
            if v["node_type"] == "leaf":
                continue
            self.parent_class_names.append(class_name)
            self.parent_syns.append(v["synonyms"])
            self.parent_class_key2num[class_key] = len(self.parent_class_names) - 1
        if self.cfg.use_synonyms_parents:
            self.parent_class_reprs: List[List[str]] = self.parent_syns
        else:
            self.parent_class_reprs: List[List[str]] = [[n] for n in self.parent_class_names]

        # setup metric for leaves
        if self.cfg.use_synonyms_leaves:
            assert self.leaf_synonym_dict is not None and len(self.leaf_synonym_dict) > 0
            self.metric = EmbeddingSynonymAccuracy(
                self.leaf_class_names,
                self.leaf_synonym_dict,
                self.cfg.arg_max_or_average_syn_leaves,
                package_name=self.cfg.package_name,
                embedder_name=self.cfg.model_name,
                templates_name=self.cfg.templates_name_leaves,
                # model_kwargs=dict(enable_vision=True),
            )
            self.leaf_class_reprs: List[List[str]] = [[]] * len(self.leaf_class_names)
            for syn_name, syn_clsidx in self.leaf_synonym_dict.items():
                self.leaf_class_reprs[syn_clsidx].append(syn_name)
        else:
            self.metric = EmbeddingAccuracy(
                self.leaf_class_names,
                package_name=self.cfg.package_name,
                embedder_name=self.cfg.model_name,
                templates_name=self.cfg.templates_name_leaves,
                # model_kwargs=dict(enable_vision=True),
            )
            self.leaf_class_reprs: List[List[str]] = [[n] for n in self.leaf_class_names]

        # use same embedder as leaf metric for everything
        self.embedder = self.metric.embedder

    def close(self):
        self.metric.close()

    def compute_sim_to_parents(self, pred_emb: torch.Tensor):
        if self.cfg.use_synonyms_parents:
            parent_syns_flat = [s for syns in self.parent_syns for s in syns]
            parent_syns_classids = [i for i, syns in enumerate(self.parent_syns) for _ in syns]
            parent_classnames_emb_syns = compute_class_embeddings_given_embedder(
                parent_syns_flat, self.embedder, self.cfg.templates_name_parents
            )
            sim_syns = cos_sim(pred_emb, parent_classnames_emb_syns)  # (50000, ~700)
            sim = reduce_synonym_logits_over_classes(
                sim_syns,
                parent_syns_classids,
                arg_max_or_average_syn=self.cfg.arg_max_or_average_syn_parents,
            )  # (50000, 393)
            return sim
        else:
            parent_classnames_emb = compute_class_embeddings_given_embedder(
                self.parent_class_names, self.embedder, self.cfg.templates_name_parents
            )
            sim = cos_sim(pred_emb, parent_classnames_emb)  # (50000, 393)
            return sim

    def compare_preds_to_leaves(self, preds: Dict[str, str]):
        targets = self.targets
        target_list = [targets[k] for k in preds.keys()]
        self.metric.reset()
        self.metric.update(list(preds.keys()), list(preds.values()), target_list)

        if isinstance(self.metric, EmbeddingSynonymAccuracy):
            sim2leaves, pred_emb, pred_synonym_ids = self.metric.compute_logits_synonyms(
                return_pred_embeddings=True
            )
        else:
            sim2leaves, pred_emb, pred_synonym_ids = self.metric.compute_logits(
                return_pred_embeddings=True
            )
        if pred_synonym_ids is not None:
            raise NotImplementedError(
                "pred_synonym_ids should be None "
                "(followup with predictions averaged over sentences not implemented.)"
            )

        labels = torch.tensor(target_list, dtype=torch.long, device=sim2leaves.device)
        logits_argmax = sim2leaves.argmax(dim=1)
        acc_tensor = torch.eq(logits_argmax, labels).float()
        logging.info(f"Accuracy: {acc_tensor.mean().item():.3%}")
        return sim2leaves, pred_emb, pred_synonym_ids, acc_tensor

    def evaluate_pipeline(self, preds: Dict[str, str], return_details_for_vis: bool = False):
        if self.cfg.cut_long_text:
            preds = {k: cut_too_long_text(v) for k, v in preds.items()}

        sim2leaves, pred_emb, pred_synonym_ids, acc_tensor = self.compare_preds_to_leaves(preds)

        norm_sim2leaves = normalize_sim_with_canonicals(
            sim2leaves,
            pred_emb,
            self.embedder,
            self.cfg.canonical_things,
            self.cfg.templates_name_leaves,
            self.cfg.canonical_mode_leaves,
        )  # (50000, 1000)

        sim2parents = self.compute_sim_to_parents(pred_emb)  # (50000, 393)
        norm_sim2parents = normalize_sim_with_canonicals(
            sim2parents,
            pred_emb,
            self.embedder,
            self.cfg.canonical_things,
            self.cfg.templates_name_parents,
            self.cfg.canonical_mode_parents,
        )
        targets = self.targets

        if self.cfg.exclude_parents is not None:
            exclude_nums = [self.parent_class_key2num[k] for k in self.cfg.exclude_parents]
            exclude_names = [self.parent_class_names[i] for i in exclude_nums]
            logging.info(
                f"Excluding parents from followup: "
                f"{exclude_names} {exclude_nums} {self.cfg.exclude_parents}"
            )
            norm_sim2parents[:, exclude_nums] = -999.0
        else:
            logging.info(f"exclude_parents not set, not excluding anything.")

        correct_answer_ids = torch.unique(torch.where(acc_tensor > 1 - 1e-4)[0]).numpy()
        incorrect_answer_ids = torch.unique(torch.where(acc_tensor <= 1 - 1e-4)[0]).numpy()
        logging.debug(f"Incorrectly answered ids: {incorrect_answer_ids.shape}")

        leaf_class_keys = self.hierarchy.get_leaf_class_keys()
        pred_keys = list(preds.keys())
        output_dict = {pred_keys[i]: {"status": "correct"} for i in correct_answer_ids}
        pbar = tqdm(total=len(incorrect_answer_ids), desc="Computing followup question")
        # print_fn = lambda *a, sep=" ", **ka: pbar.write(sep.join(str(x) for x in a), **ka)
        print_fn = lambda *_a, **_ka: None
        # reroute_logger(lambda msg: pbar.write(msg, end=""))
        missing_parent_keys = []
        for i, pred_i in enumerate(incorrect_answer_ids):
            output_dict_here = {}
            pred_key = pred_keys[pred_i]
            pred_val = preds[pred_key]
            target = targets[pred_key]
            leaf_class_name = self.leaf_class_names[target]
            leaf_class_key = leaf_class_keys[target]
            parent_class_keys = self.hierarchy.get_parent_keys(leaf_class_key)
            if return_details_for_vis:
                output_dict_here.update(
                    {
                        "leaf_class_name": leaf_class_name,
                        "sim2parents": sim2parents[pred_i],
                        "sim2leaves": sim2leaves[pred_i],
                        "norm_sim2parents": norm_sim2parents[pred_i],
                        "norm_sim2leaves": norm_sim2leaves[pred_i],
                        "pred_val": pred_val,
                        "parent_class_keys": parent_class_keys,
                    }
                )

            # calculate followup question
            parent_class_nums = [
                self.parent_class_key2num[key]
                for key in parent_class_keys
                if key in self.parent_class_key2num
            ]
            # print missing keys as a warning
            if len(parent_class_nums) != len(parent_class_keys):
                missing_parent_keys += [
                    key for key in parent_class_keys if key not in self.parent_class_key2num
                ]
            norm_sim2p_hierarchy = norm_sim2parents[pred_i, parent_class_nums]
            closest_hier_val, closest_hier_idx = torch.max(norm_sim2p_hierarchy, dim=0)
            if closest_hier_val < self.cfg.threshold:
                output_dict_here["status"] = "failed"
                print_fn(f"Failed!")
            else:
                closest_hier_key = parent_class_keys[closest_hier_idx.item()]
                closest_hier_num = self.parent_class_key2num[closest_hier_key]
                closest_hier_name = self.parent_class_names[closest_hier_num]
                print_fn(f"Ask about: {closest_hier_name}")
                output_dict_here.update(
                    {"status": "followup", self.cfg.default_followup_object: closest_hier_name}
                )
            output_dict[pred_key] = output_dict_here
            pbar.update()
        pbar.close()

        # print warning about missing keys with parents names
        if len(missing_parent_keys) > 0:
            missing_parent_keys = list(set(missing_parent_keys))
            logging.warning(f"Missing parent -> if other than root or a leaf, this is a bug.")
            logging.warning(
                f"Missing parent: size {len(missing_parent_keys)}, {[(key, self.hierarchy.get_class(key)['class_name']) for key in missing_parent_keys]}"
            )
        # reroute_logger(new_sink=sys.stderr)
        return output_dict


def visualize_table(
    table_data: List[List[Any]], table_highlight: List[bool], print_fn=print, disable=False
):
    if disable:
        return
    for row, highlight in zip(table_data, table_highlight):
        print_fn(f"{'*' if highlight else ' '}", *row)


def normalize_sim_with_canonicals(
    sim,  # similarity from prediction to target shape (num_pred, num_target)
    pred_emb,  # prediction embeddings shape (num_pred, emb_dim)
    embedder: SentenceEmbedderInterface,
    canonical_things: List[str],  # canonicals
    templates_name: str,  # template prompt
    canonical_mode: str = "min",
):
    if canonical_mode is None or canonical_mode == "none":
        return sim
    canonical_emb = compute_class_embeddings_given_embedder(
        canonical_things, embedder, templates_name
    )
    canonical_sim = cos_sim(pred_emb, canonical_emb)  # (50000, 4)
    canons = torch.exp(sim[:, :, None]) / (
        torch.exp(canonical_sim[:, None, :]) + torch.exp(sim[:, :, None])
    )  # (50000, 393, n_canons)
    if canonical_mode == "avg":
        norm_sim = canons.mean(dim=2)
    elif canonical_mode == "min":
        norm_sim, _ = torch.min(
            canons,
            dim=2,
        )
    else:
        raise ValueError(f"Unknown canonical mode: {canonical_mode}")
    return norm_sim


def fmt_sim(sim):
    return f"{float(sim) * 1000:4.0f}"


def main():
    followup_cfg = FollowupConfig(
        templates_name_parents="openai_imagenet_template",
        threshold="0.53",
    )
    hier = ImagenetHierarchy.load()

    class_names = [
        d["clip_bench_label"]
        for d in load_json(
            get_ovqa_annotations_dir() / "imagenet1k/class_hierarchy/labels_data.json"
        )
    ]
    follower = Followup(
        cfg=followup_cfg,
        hierarchy=hier,
        leaf_class_names=class_names,
        leaf_synonym_dict=None,
        targets=None,
    )


if __name__ == "__main__":
    main()
