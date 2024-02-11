import torch
from omegaconf import OmegaConf
from transformers import CLIPImageProcessor

from ovqa.common.lavis.registry import registry
from ovqa.processors.base_processor import BaseProcessor


@registry.register_processor("llava_text")
class LlavaTextProcessor(BaseProcessor):
    """
    For simplicity leave all text processing inside the model itself.
    """

    def __call__(self, caption):
        return caption


@registry.register_processor("llava_image_eval")
class LlavaImageEvalProcessor(BaseProcessor):
    def __init__(self, image_size=224):
        assert image_size == 224
        super().__init__()

        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=torch.float16
        )

    def __call__(self, item):
        return self.image_processor.preprocess(item, return_tensors="pt")["pixel_values"][0]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)
        return cls(image_size=image_size)
