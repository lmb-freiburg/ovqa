from omegaconf import OmegaConf
from open_clip import get_pretrained_cfg, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD, image_transform

from ovqa.common.lavis.registry import registry
from ovqa.processors.base_processor import BaseProcessor


@registry.register_processor("openclip_image_eval")
class OpenClipImageEvalProcessor(BaseProcessor):
    def __init__(self, cfg):
        super().__init__()
        model_name = cfg["model_name"]
        pretrained_name = cfg["pretrained_name"]
        image_size = cfg.get("image_size", 224)
        pretrained_cfg = get_pretrained_cfg(model_name, pretrained_name)

        # get image mean and std either from this lavis config, or open clip config, or default
        mean = cfg.get("mean", pretrained_cfg.get("mean", OPENAI_DATASET_MEAN))
        std = cfg.get("std", pretrained_cfg.get("std", OPENAI_DATASET_STD))
        self.transform = image_transform(image_size, is_train=False, mean=mean, std=std)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        return cls(cfg)


@registry.register_processor("openclip_text")
class OpenClipTextEvalProcessor(BaseProcessor):
    """Assuming the classnames are already in OK format, we do not need additional preprocessing"""

    def __init__(self, _cfg):
        super().__init__()

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        return cls(cfg)
