from collections import defaultdict

import numpy as np
import torch
from sentence_transformers.util import cos_sim
from typing import List, Dict

from ovqa.metrics.clip_match import (
    compute_class_embeddings,
    EmbeddingAccuracy,
    get_templates,
    compute_class_embeddings_given_embedder,
)


class EmbeddingSynonymAccuracy(EmbeddingAccuracy):
    def __init__(
        self,
        class_names: List[str],
        synonym_dict: Dict[str, int] = None,  # dictionary of synonyms to class indices
        arg_max_or_average_syn: str = "arg_max_syn",
        **kwargs,
    ) -> None:
        super().__init__(class_names, **kwargs)
        if synonym_dict is None:
            synonym_dict = {}

        assert arg_max_or_average_syn in [
            "arg_max_syn",
            "average_syn",
        ], f"Definition of synonym behavior {arg_max_or_average_syn} not defined, selectbetween [arg_max_syn, average_syn]"

        self.arg_max_or_average_syn = arg_max_or_average_syn
        if len(synonym_dict) > 0:
            self.synonym_names = list(synonym_dict.keys())
            self.synonym2clsidx = list(synonym_dict.values())
            self.synonym_embeddings = torch.from_numpy(
                compute_class_embeddings(
                    self.synonym_names,
                    self.embedder_name,
                    self.package_name,
                    self.templates_name,
                    model_kwargs=kwargs.get("model_kwargs", None),
                )
            )
        else:
            self.synonym_names = self.class_names
            self.synonym2clsidx = list(range(len(self.class_names)))
            self.synonym_embeddings = self.class_embeddings

    def compute_logits_synonyms(self, return_pred_embeddings: bool = False) -> torch.Tensor:
        """Compute the metric. Have as argument synonym_embeddings for computing the logits"""
        logits_synonyms, pred_embeddings, pred_syn_ids = self.compute_logits(
            self.synonym_embeddings, return_pred_embeddings=True
        )

        logits = []
        for indx_cls in range(len(self.class_names)):
            syn_index = []
            # since clsidx could be a list then we go through it
            for indx, values in enumerate(self.synonym2clsidx):
                if isinstance(values, int):
                    values = [values]
                if indx_cls in set(values):
                    syn_index.append(indx)

            if len(syn_index) == 0:
                logits.append(
                    -1e4 * torch.ones(logits_synonyms.shape[0], device=logits_synonyms.device)
                )
            if self.arg_max_or_average_syn == "average_syn":
                logits.append(logits_synonyms[:, syn_index].mean(1))
            elif self.arg_max_or_average_syn == "arg_max_syn":
                val, _ = logits_synonyms[:, syn_index].max(1)
                logits.append(val)
            else:
                # throw an exception error not defined
                assert (
                    False
                ), f"Definition of synonym behavior {self.arg_max_or_average_syn} not defined, selectbetween [arg_max_syn, average_syn]"

        logits = torch.stack(logits, dim=1)
        if return_pred_embeddings:
            return logits, pred_embeddings, pred_syn_ids
        return logits

    def compute(self) -> float:
        """Compute the metric."""
        logits = self.compute_logits_synonyms()
        softmaxed_logits = torch.softmax(logits, dim=1)
        labels = torch.tensor(self.target_ids, dtype=torch.long, device=logits.device)
        acc = self.accuracy_module(softmaxed_logits, labels)
        return acc.item()


class EmbeddingSynonymScore(EmbeddingSynonymAccuracy):
    def __init__(
        self,
        class_names: List[str],
        **kwargs,
    ) -> None:
        super().__init__(class_names, **kwargs)
        self.compute_threshold_sim_per_class()

    def compute(self, return_per_cls=False, retun_logits=False) -> float:
        """Compute the metric."""
        logits = self.compute_logits_synonyms()
        target_ids_tensor = torch.tensor(self.target_ids, dtype=torch.long, device=logits.device)
        logits_pos = logits[torch.arange(target_ids_tensor.shape[0]), target_ids_tensor]

        acc = []
        acc_per_cls = []
        for cls_idx in range(len(self.class_names)):
            cls_acc = (
                logits_pos[torch.eq(torch.Tensor(self.target_ids), cls_idx)]
                > self.sim_stat_per_class[cls_idx]["threshold"]
            ).to(dtype=torch.float)
            acc.append(cls_acc)
            acc_per_cls.append(cls_acc.nanmean())
        acc = torch.cat(acc, dim=0).nanmean()
        acc_per_cls = torch.Tensor(acc_per_cls)
        mAcc = acc_per_cls.nanmean().item()

        to_return = (acc.item(),)
        if return_per_cls:
            to_return += ((mAcc, acc_per_cls),)
            # return acc, (mAcc, acc_per_cls)
        if retun_logits:
            to_return += (logits_pos,)

        if len(to_return) == 1:
            return to_return[0]
        return to_return

    def compute_threshold_sim_per_class(self, template_to_compare="answers_v0"):
        # compute the similarity among the correct answers using different prefixes in answers tamplates
        answers_template = get_templates(template_to_compare)
        prompted_answers = [p.format(c=syn) for syn in self.synonym_names for p in answers_template]
        syn_prompted_embeddings = torch.Tensor(
            compute_class_embeddings_given_embedder(
                prompted_answers,
                self.embedder,
                templates_name="none",
                use_cache=True,
            )
        )
        syn_prompted_embeddings = torch.split(syn_prompted_embeddings, len(answers_template))
        synonym_embeddings = self.synonym_embeddings
        sim_stat_per_class = defaultdict(dict)
        # compute the raw synonyms similarity
        for indx_cls in range(len(self.class_names)):
            # print(self.class_names[indx_cls])
            syn_index = []
            # since clsidx is a list then we go through it
            for indx, values in enumerate(self.synonym2clsidx):
                if isinstance(values, int):
                    values = [values]
                if indx_cls in set(values):
                    syn_index.append(indx)
            syn_index = list(set(syn_index))
            if len(syn_index) > 1:
                syn_emb = synonym_embeddings[syn_index]
                (
                    upper_triangle,
                    min_value,
                    mean_value,
                    std_value,
                    max_value,
                ) = compute_similarity_and_statistics_given_embeddings(syn_emb)
                sim_stat_per_class[indx_cls]["synonyms"] = {
                    # "uptriangle": upper_triangle,
                    "min": min_value.item(),
                    "avg": mean_value.item(),
                    "std": std_value.item(),
                    "max": max_value.item(),
                }

            # calculate similarity stat across possible answers
            dict_stat_class = defaultdict(list)
            for syn_idx in syn_index:
                syn_emb = torch.cat(
                    [synonym_embeddings[syn_idx : syn_idx + 1], syn_prompted_embeddings[syn_idx]],
                    axis=0,
                )
                (
                    upper_triangle,
                    min_value,
                    mean_value,
                    std_value,
                    max_value,
                ) = compute_similarity_and_statistics_given_embeddings(syn_emb)
                # save statistics
                dict_stat_class["min"].append(min_value.item())
                dict_stat_class["avg"].append(mean_value.item())
                dict_stat_class["std"].append(std_value.item())
                dict_stat_class["max"].append(max_value.item())

                # print(self.synonym_names[syn_idx], [prompted_answers[pa] for pa in range(len(answers_template)*syn_idx, len(answers_template)*(syn_idx+1))])
                # print(upper_triangle)

            sim_stat_per_class[indx_cls]["sentence"] = {
                "min": np.asarray(dict_stat_class["min"]).mean(),
                "avg": np.asarray(dict_stat_class["avg"]).mean(),
                "std": np.asarray(dict_stat_class["std"]).mean(),
                "max": np.asarray(dict_stat_class["max"]).mean(),
            }

            th = min(
                sim_stat_per_class[indx_cls].get("synonyms", {}).get("avg", 100),
                sim_stat_per_class[indx_cls].get("sentence", {}).get("min", 100),
            )
            sim_stat_per_class[indx_cls]["threshold"] = th

        self.sim_stat_per_class = sim_stat_per_class


def compute_similarity_and_statistics_given_embeddings(embeddings):
    matrix = cos_sim(embeddings, embeddings)

    # Exclude the diagonal values by filling them with a large negative value
    matrix = matrix - torch.diag(torch.diag(matrix))

    # Get the upper triangle elements using torch.triu
    upper_triangle = torch.triu(matrix, diagonal=1)
    # print(upper_triangle)

    # Calculate the min, mean, std, and max of the upper triangle
    pos_values = upper_triangle[upper_triangle > 0]
    min_value = torch.min(pos_values[pos_values < 1])
    mean_value = torch.mean(pos_values[pos_values < 1])
    std_value = torch.std(pos_values[pos_values < 1])
    max_value = torch.max(pos_values[pos_values < 1])

    return upper_triangle, min_value, mean_value, std_value, max_value


def main():
    labels = "person", "dog", "cat"
    captions = (
        "a girl in a green dress on a motorcycle",
        "a man water skiing",
        "an image of a black cat",
        "a dog near a black house",
    )
    synonyms_dict = {
        "person": 0,
        "human": 0,
        "man": 0,
        "woman": 0,
        "girl": 0,
        "boy": 0,
        "kid": 0,
        "child": 0,
        "baby": 0,
        "player": 0,
        "dog": 1,
        "white dog": 1,
        "white puppy": 1,
        "light dog": 1,
        "light puppy": 1,
        "cat": 2,
        "black cat": 2,
        "black kitten": 2,
        "dark cat": 2,
        "dark kitten": 2,
    }
    metric = EmbeddingSynonymAccuracy(class_names=labels, synonym_dict=synonyms_dict)
    metric.update(["1", "2", "3", "4"], captions, [0, 0, 2, 1])
    print(metric.compute())


if __name__ == "__main__":
    main()
