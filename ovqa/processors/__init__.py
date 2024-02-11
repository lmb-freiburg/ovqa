"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from ovqa.common.lavis.registry import registry
from ovqa.processors.base_processor import BaseProcessor
from ovqa.processors.blip_processors import (
    BlipImageTrainProcessor,
    Blip2ImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)
from ovqa.processors.clip_processors import ClipImageTrainProcessor
from ovqa.processors.gpt_processors import GPTVideoFeatureProcessor, GPTDialogueProcessor
from ovqa.processors.llava_processors import LlavaImageEvalProcessor, LlavaTextProcessor
from ovqa.processors.openclip_processors import (
    OpenClipImageEvalProcessor,
    OpenClipTextEvalProcessor,
)
from ovqa.processors.text_none_processor import TextNoneProcessor
from ovqa.processors.x2vlm_processors import X2vlmImageEvalProcessor, X2vlmTextProcessor

__all__ = [
    "BaseProcessor",
    "BlipImageTrainProcessor",
    "Blip2ImageTrainProcessor",
    "BlipImageEvalProcessor",
    "BlipCaptionProcessor",
    "ClipImageTrainProcessor",
    "GPTVideoFeatureProcessor",
    "GPTDialogueProcessor",
    "OpenClipImageEvalProcessor",
    "OpenClipTextEvalProcessor",
    "LlavaImageEvalProcessor",
    "LlavaTextProcessor",
    "TextNoneProcessor",
    "X2vlmImageEvalProcessor",
    "X2vlmTextProcessor",
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
