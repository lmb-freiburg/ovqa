"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import torch
from transformers import BertTokenizer

from ovqa.common.lavis.dist_utils import download_cached_file
from ovqa.common.lavis.utils import is_url
from ovqa.models.lavis.base_model import BaseModel
from ovqa.models.lavis.vit import interpolate_pos_embed


class BlipBase(BaseModel):
    def __init__(self):
        super().__init__()
        # # the fix is simply to not repeat things when num_beams > 1
        # transformers_version = version.parse(transformers.__version__)
        # assert transformers_version < version.parse(
        #     "4.27"
        # ), "BLIP models are not compatible with transformers>=4.27, run pip install transformers==4.25 to downgrade"

    @classmethod
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        return tokenizer

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], self.visual_encoder
        )
        if "visual_encoder_m.pos_embed" in self.state_dict().keys():
            if "visual_encoder_m.pos_embed" not in state_dict.keys():
                logging.warning("visual_encoder_m.pos_embed not in checkpoint!")
            else:
                state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
                    state_dict["visual_encoder_m.pos_embed"], self.visual_encoder_m
                )

        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg
