import hashlib
from typing import List, Callable

import numpy as np
import torch
from loguru import logger
from sentence_transformers.util import cos_sim
from torchmetrics import Accuracy

from ovqa.annotations.cls_templates import CLASSIFICATION_TEMPLATES
from ovqa.metrics.torchmetrics_ext import MetricExt
from ovqa.textutils.embeddings import (
    get_sentence_embedder,
    EmbeddingsPackageConst,
    SentenceEmbedderInterface,
)
from ovqa.paths import get_cache_dir

EOS = "."


def get_templates(templates_name="none") -> List[Callable]:
    if templates_name == "imagenet1k":
        print(f"WARN: template name imagenet1k is deprecated, use openai_imagenet_template instead")
        templates_name = "openai_imagenet_template"
    return CLASSIFICATION_TEMPLATES[templates_name]


def compute_class_embeddings(
    class_names: List[str],
    model_name: str,
    package_name: str,
    templates_name: str,
    model_kwargs=None,
):
    model_kwargs = model_kwargs if model_kwargs is not None else {}
    embedder = get_sentence_embedder(
        model_name=model_name, package_name=package_name, **model_kwargs
    )
    return compute_class_embeddings_given_embedder(class_names, embedder, templates_name)


def compute_class_embeddings_given_embedder(
    class_names: List[str],
    embedder: SentenceEmbedderInterface,
    templates_name: str,
    use_cache: bool = True,
):
    model_name, package_name = embedder.model_name, embedder.package_name

    # load from cache if exists
    classes_hsh = hashlib.sha3_256(repr(class_names).encode()).hexdigest()
    cache_file = (
        get_cache_dir()
        / "class_embeddings"
        / f"{package_name}_{model_name.replace('/','--')}_{templates_name}_{classes_hsh}.npy"
    )
    if use_cache and cache_file.is_file():
        data = np.load(cache_file)
        logger.debug(f"Loaded class embeddings shape {data.shape} from {cache_file}")
        return data

    # create class embeddings
    logger.debug(
        f"Compute class embeddings for {len(class_names)} classes with {model_name} "
        f"and {templates_name} templates"
    )
    prompts_list = get_templates(templates_name)
    prompts_classes = [p(c=class_name) for class_name in class_names for p in prompts_list]
    n_prompts = len(prompts_list)
    n_classes = len(class_names)
    embeddings = embedder.encode(prompts_classes, normalize=True)
    emb_re = np.reshape(embeddings, (n_classes, n_prompts, -1))
    emb_mean = np.mean(emb_re, axis=1)
    cache_file.parent.mkdir(exist_ok=True, parents=True)
    np.save(cache_file, emb_mean)
    return emb_mean


def default_preprocess_class_name(class_name: str) -> str:
    return class_name.replace("_", " ")


class EmbeddingAccuracy(MetricExt):
    def __init__(
        self,
        class_names: list[str],
        package_name=EmbeddingsPackageConst.OPEN_CLIP,
        embedder_name="EVA01-g-14/laion400m_s11b_b41k",
        templates_name="openai_imagenet_template",
        top_k=1,
        preprocess_class_name_fn: Callable = default_preprocess_class_name,
        preprocess_prediction_fn: Callable = None,
        model_kwargs=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("pred_keys", default=[], dist_reduce_fx="cat")
        self.add_state("pred_values", default=[], dist_reduce_fx="cat")
        self.add_state("target_ids", default=[], dist_reduce_fx="cat")

        self.accuracy_module = Accuracy(
            task="multiclass", num_classes=len(class_names), top_k=top_k
        )
        self.accuracy_module_per_class = Accuracy(
            task="multiclass", num_classes=len(class_names), top_k=top_k, average="none"
        )
        self.top_k = top_k

        if preprocess_class_name_fn is not None:
            class_names = [preprocess_class_name_fn(c) for c in class_names]
        self.class_names = class_names
        self.package_name = package_name
        self.embedder_name = embedder_name
        self.templates_name = templates_name

        self.preprocess_prediction_fn = preprocess_prediction_fn
        model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.embedder = get_sentence_embedder(
            model_name=embedder_name, package_name=package_name, **model_kwargs
        )
        self.class_embeddings = torch.from_numpy(
            compute_class_embeddings_given_embedder(
                self.class_names, self.embedder, self.templates_name
            )
        )

    def update(self, keys: List[str], values: List[str], target_ids: List[int]) -> None:
        """Update state with data.

        Args:
            keys (list[str]): Batch of keys to identify the datapoint.
            values (list[str]): Batch of model predictions.
            target_ids (list[str]): Batch of target class indices.
        """
        self.pred_keys.extend(keys)
        self.pred_values.extend(values)
        self.target_ids.extend(target_ids)

    def compute(self) -> float:
        """Compute the metric."""
        logits = self.compute_logits()
        softmaxed_logits = torch.softmax(logits, dim=1)
        labels = torch.tensor(self.target_ids, dtype=torch.long, device=logits.device)
        acc = self.accuracy_module(softmaxed_logits, labels)
        return acc.item()

    def compute_logits(
        self, target_embeddings=None, return_pred_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Returns:
            logits: shape (n_preds, n_classes)
            embeddings: shape (n_preds, emb_dim) only if return_embeddings is True
            synonym_ids: if synonyms are used, returns the ids of the synonyms
                e.g. [0, 0, 0, 0, 1, 1, 2, ...]
        """
        if target_embeddings is None:
            target_embeddings = self.class_embeddings
        return compute_logits(
            target_embeddings,
            self.embedder,
            self.pred_values,
            self.preprocess_prediction_fn,
            return_pred_embeddings=return_pred_embeddings,
        )

    def compute_per_datapoint(self, return_dict=False) -> torch.Tensor:
        logits = self.compute_logits()
        softmaxed_logits = torch.softmax(logits, dim=1)
        labels = torch.tensor(self.target_ids, dtype=torch.long, device=logits.device)
        if self.top_k == 1:
            pred_ids = torch.argmax(softmaxed_logits, dim=1)
            acc = pred_ids == labels
        else:
            _, pred_ids = torch.topk(softmaxed_logits, k=self.top_k, dim=1)
            acc = torch.any(pred_ids == labels.unsqueeze(1), dim=1)
        scores = acc.float()
        if return_dict:
            return {"scores": scores}
        return scores

    def compute_per_class(self) -> torch.Tensor:
        logits = self.compute_logits()
        softmaxed_logits = torch.softmax(logits, dim=1)
        labels = torch.tensor(self.target_ids, dtype=torch.long, device=logits.device)
        acc = self.accuracy_module_per_class(softmaxed_logits, labels)
        return acc

    def close(self):
        self.embedder.close()


def compute_logits(
    target_embeddings,
    embedder,
    preds: list[str],
    preprocess_prediction_fn=None,
    return_pred_embeddings: bool = False,
) -> torch.Tensor:
    """
    Returns:
        logits: shape (n_preds, n_classes)
    if return_embeddings is True, also returns:
        embeddings: shape (n_preds, emb_dim) only
        synonym_ids: if synonyms are used, returns the ids of the synonyms
            e.g. [0, 0, 0, 0, 1, 1, 2, ...]
    """
    assert target_embeddings is not None, f"target_embeddings must not be None"
    assert not isinstance(target_embeddings[0], str), (
        f"target_embeddings must be a list of embeddings, "
        f"not a list of strings. Got {target_embeddings[0]} as first element"
    )

    embedder = embedder
    pred_values = preds
    if preprocess_prediction_fn is not None:
        pred_values = [preprocess_prediction_fn(v) for v in pred_values]

    pred_embeddings = embedder.encode(pred_values, normalize=True)
    logits = cos_sim(pred_embeddings, target_embeddings)
    if return_pred_embeddings:
        return logits, pred_embeddings, None
    return logits


def main():
    labels = "black cats", "white dog", "black dog", "grouse", "hare", "skateboard"
    captions = (
        "an image of a black cat",
        "a dog near a black house",
        "the 7 seas",
        "elvis presley",
    )
    metric = EmbeddingAccuracy(class_names=labels)
    metric.update(["1", "2", "3", "4"], captions, [0, 1, 2, 3])
    print(metric.compute())


if __name__ == "__main__":
    main()
