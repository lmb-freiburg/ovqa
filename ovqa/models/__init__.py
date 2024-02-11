"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
from omegaconf import OmegaConf

from ovqa.common.lavis.registry import registry
from ovqa.models.lavis.base_model import BaseModel
from ovqa.models.lavis.blip2_models.blip2 import Blip2Base
from ovqa.models.lavis.blip2_models.blip2_image_text_matching import Blip2ITM
from ovqa.models.lavis.blip2_models.blip2_opt import Blip2OPT
from ovqa.models.lavis.blip2_models.blip2_qformer import Blip2Qformer
from ovqa.models.lavis.blip2_models.blip2_t5 import Blip2T5
from ovqa.models.lavis.blip2_models.blip2_t5_instruct import Blip2T5Instruct
from ovqa.models.lavis.blip2_models.blip2_vicuna_instruct import Blip2VicunaInstruct
from ovqa.models.lavis.blip_models.blip import BlipBase
from ovqa.models.lavis.blip_models.blip_caption import BlipCaption
from ovqa.models.lavis.blip_models.blip_classification import BlipClassification
from ovqa.models.lavis.blip_models.blip_feature_extractor import BlipFeatureExtractor
from ovqa.models.lavis.blip_models.blip_image_text_matching import BlipITM
from ovqa.models.lavis.blip_models.blip_nlvr import BlipNLVR
from ovqa.models.lavis.blip_models.blip_pretrain import BlipPretrain
from ovqa.models.lavis.blip_models.blip_retrieval import BlipRetrieval
from ovqa.models.lavis.blip_models.blip_vqa import BlipVQA
from ovqa.models.lavis.clip_models.model import CLIP
from ovqa.models.lavis.gpt_models.gpt_dialogue import GPTDialogue
from ovqa.models.lavis.img2prompt_models.img2prompt_vqa import Img2PromptVQA
from ovqa.models.lavis.med import XBertLMHeadDecoder
from ovqa.models.lavis.pnp_vqa_models.pnp_unifiedqav2_fid import PNPUnifiedQAv2FiD
from ovqa.models.lavis.pnp_vqa_models.pnp_vqa import PNPVQA
from ovqa.models.lavis.vit import VisionTransformerEncoder
from ovqa.models.llava_model import LlavaLavis
from ovqa.models.open_clip_model import OpenClipModel
from ovqa.models.x2vlm_model import X2vlmLavis
from ovqa.processors.base_processor import BaseProcessor
from ovqa.processors.llava_processors import LlavaTextProcessor, LlavaImageEvalProcessor
from ovqa.processors.text_none_processor import TextNoneProcessor
from ovqa.processors.x2vlm_processors import X2vlmTextProcessor, X2vlmImageEvalProcessor


def load_model(name, model_type, is_eval=False, device="cpu", checkpoint=None):
    """
    Load supported models.

    To list all available models and types in registry:
    >>> from ovqa.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu". None = do not modify device.
        checkpoint (str): path or to checkpoint. Default: None.
            Note that expecting the checkpoint to have the same keys in state_dict as the model.

    Returns:
        model (torch.nn.Module): model.
    """

    model = registry.get_model_class(name).from_pretrained(model_type=model_type)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

    if is_eval:
        model.eval()

    if device is None:
        return model

    if device == "cpu":
        model = model.float()

    return model.to(device)


def load_preprocess(config):
    """
    Load preprocessor configs and construct preprocessors.

    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

    Args:
        config (dict): preprocessor configs.

    Returns:
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.

        Key is "train" or "eval" for processors used in training and evaluation respectively.
    """

    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else BaseProcessor()
        )

    vis_processors = dict()
    txt_processors = dict()

    vis_proc_cfg = config.get("vis_processor")
    txt_proc_cfg = config.get("text_processor")

    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get("train")
        vis_eval_cfg = vis_proc_cfg.get("eval")
    else:
        vis_train_cfg = None
        vis_eval_cfg = None

    vis_processors["train"] = _build_proc_from_cfg(vis_train_cfg)
    vis_processors["eval"] = _build_proc_from_cfg(vis_eval_cfg)

    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get("train")
        txt_eval_cfg = txt_proc_cfg.get("eval")
    else:
        txt_train_cfg = None
        txt_eval_cfg = None

    txt_processors["train"] = _build_proc_from_cfg(txt_train_cfg)
    txt_processors["eval"] = _build_proc_from_cfg(txt_eval_cfg)

    return vis_processors, txt_processors


def load_model_and_preprocess(name, model_type, is_eval=False):
    """
    Load model and its related preprocessors.
    List all available models and types in registry:
    >>> from ovqa.models import model_zoo
    >>> print(model_zoo)
    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
    Returns:
        model (torch.nn.Module): model.
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.
    """
    model_cls = registry.get_model_class(name)
    model = model_cls.from_pretrained(model_type=model_type)
    if is_eval:
        model.eval()
    # load preprocess
    config_path = model_cls.default_config_path(model_type)
    logging.info(f"Loading config for model {name} type {model_type} from {config_path}")
    cfg = OmegaConf.load(config_path)
    if cfg is not None:
        preprocess_cfg = cfg.preprocess
        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )
    return model, vis_processors, txt_processors


class ModelZoo:
    """
    A utility class to create string representation of available model architectures and types.

    >>> from ovqa.models import model_zoo
    >>> # list all available models
    >>> print(model_zoo)
    >>> # show total number of models
    >>> print(len(model_zoo))
    """

    def __init__(self) -> None:
        self.model_zoo = {
            k: list(v.PRETRAINED_MODEL_CONFIG_DICT.keys())
            for k, v in registry.mapping["model_name_mapping"].items()
        }

    def __str__(self) -> str:
        return (
            "=" * 50
            + "\n"
            + f"{'Architectures':<30} {'Types'}\n"
            + "=" * 50
            + "\n"
            + "\n".join(
                [f"{name:<30} {', '.join(types)}" for name, types in self.model_zoo.items()]
            )
        )

    def __iter__(self):
        return iter(self.model_zoo.items())

    def __len__(self):
        return sum([len(v) for v in self.model_zoo.values()])


model_zoo = ModelZoo()

__all__ = [
    "load_model",
    "load_model_and_preprocess",
    "model_zoo",
    "BaseModel",
    "BlipBase",
    "BlipFeatureExtractor",
    "BlipCaption",
    "BlipClassification",
    "BlipITM",
    "BlipNLVR",
    "BlipPretrain",
    "BlipRetrieval",
    "BlipVQA",
    "Blip2Qformer",
    "Blip2Base",
    "Blip2ITM",
    "Blip2OPT",
    "Blip2T5",
    "Blip2T5Instruct",
    "Blip2VicunaInstruct",
    "PNPVQA",
    "Img2PromptVQA",
    "PNPUnifiedQAv2FiD",
    "CLIP",
    "VisionTransformerEncoder",
    "XBertLMHeadDecoder",
    "GPTDialogue",
    "OpenClipModel",
    "LlavaLavis",
    "LlavaTextProcessor",
    "LlavaImageEvalProcessor",
    "X2vlmLavis",
    "X2vlmTextProcessor",
    "X2vlmImageEvalProcessor",
    "TextNoneProcessor",
]
