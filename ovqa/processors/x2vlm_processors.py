from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

from ovqa.common.lavis.registry import registry
from ovqa.processors.base_processor import BaseProcessor


@registry.register_processor("x2vlm_text")
class X2vlmTextProcessor(BaseProcessor):
    """
    For simplicity leave all text processing inside the model itself.
    """

    def __call__(self, caption):
        return caption


@registry.register_processor("x2vlm_image_eval")
class X2vlmImageEvalProcessor(BaseProcessor):
    def __init__(self, image_size=224):
        super().__init__()

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ]
        )

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)
        return cls(image_size=image_size)
