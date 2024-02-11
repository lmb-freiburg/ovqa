import contextlib
import logging
import torch
from open_clip import get_tokenizer, create_model
from pathlib import Path
from torch.nn import functional as F
from tqdm import tqdm
from typing import Callable

from ovqa.common.lavis.registry import registry
from ovqa.models.lavis.base_model import BaseModel
from ovqa.tasks.multimodal_classification import (
    MultimodalClassificationTask,
    MultimodalClassificationSynonymsTask,
    get_classnames_for_classifier,
    get_classifier_cache_file,
    get_classnames_templates_for_classifier,
)


@registry.register_model("openclip")
class OpenClipModel(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {"default": "ovqa/configs/models/openclip.yaml"}

    @classmethod
    def from_config(cls, cfg):
        model_name = cfg["model_name"]
        pretrained_name = cfg["pretrained_name"]
        img_size = cfg.get("image_size", 224)
        return cls(model_name, pretrained_name, img_size)

    def __init__(
        self,
        model_name="EVA01-g-14",
        pretrained_name="laion400m_s11b_b41k",
        img_size=224,
        use_float16=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained_name = pretrained_name
        self.img_size = img_size
        self.use_float16 = use_float16

        self.loaded_ckpt_path = f"openclip/{self.model_name}/{self.pretrained_name}"
        self.model = create_model(self.model_name, self.pretrained_name)
        self.tokenizer = get_tokenizer(self.model_name)

    def maybe_autocast_float16(self):
        if self.device == torch.device("cpu") or not self.use_float16:
            # if on cpu, don't use autocast
            return contextlib.nullcontext()
        return torch.cuda.amp.autocast(dtype=torch.float16)

    @torch.no_grad()
    def before_evaluation(self, dataset, task_type, **kwargs):
        if task_type not in {MultimodalClassificationTask, MultimodalClassificationSynonymsTask}:
            return
        classnames = get_classnames_for_classifier(dataset, task_type)
        self.classifier_classnames = classnames
        templates = get_classnames_templates_for_classifier(dataset)

        cache_file = get_classifier_cache_file(classnames, templates, self)
        if cache_file.is_file():
            logging.info(f"Load classifier weights from {cache_file}")
            self.classifier = torch.load(cache_file, map_location=self.device)
            return
        logging.info(f"Create zeroshot classifier for {len(classnames)} classes")
        with self.maybe_autocast_float16():
            self.classifier = zero_shot_classifier(
                self.model, self.tokenizer, classnames, templates, self.device
            )
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.classifier, cache_file)

    @torch.no_grad()
    def predict(self, samples, return_embedding=False):
        """Zero-shot classification"""
        image = samples["image"]
        targets = samples["label"]

        with self.maybe_autocast_float16():
            image_features = self.model.encode_image(image)
            image_features = F.normalize(image_features, dim=-1)
            logits = 100.0 * image_features @ self.classifier
        output = {"predictions": logits, "targets": targets}
        if return_embedding:
            output["embeddings"] = image_features
        return output


def zero_shot_classifier(model, tokenizer, classnames, templates, device):
    zeroshot_weights = []
    for classname in tqdm(classnames):
        # templates can be a format string or a function
        if isinstance(templates[0], Callable):
            texts = [template(classname) for template in templates]
        elif isinstance(templates[0], str):
            texts = [template.format(c=classname) for template in templates]
        else:
            raise ValueError(f"Invalid template type {type(templates[0])}")
        texts = tokenizer(texts).to(device)  # tokenize
        class_embeddings = model.encode_text(texts)
        class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights
