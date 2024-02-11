"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import ovqa.common.lavis.dist_utils as dist_utils
from ovqa.common.lavis.dist_utils import download_cached_file
from ovqa.common.lavis.utils import is_url
from ovqa.common.lavis.logger import MetricLogger
from ovqa.models.lavis.base_model import BaseModel
from ovqa.models.lavis.blip2_models.Qformer import BertConfig, BertLMHeadModel
from ovqa.models.lavis.eva_vit import create_eva_vit_g
from ovqa.models.lavis.clip_vit import create_clip_vit_L
from transformers import BertTokenizer


class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", truncation_side=truncation_side
        )
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        force_autocast = getattr(self, "force_autocast", False)

        # if num_gpus == 2, device will be cpu but in reality the model weights are
        # split on 2 gpus. in that case, use autocast still
        num_gpus = getattr(self, "num_gpus", 1)
        is_cpu = self.device == torch.device("cpu") and num_gpus <= 1

        if is_cpu and not force_autocast:
            # if on cpu, don't use autocast
            return contextlib.nullcontext()

        # if on gpu, special handling of bfloat16 from config
        if dtype == torch.bfloat16:
            if hasattr(self, "bfloat16_dtype"):
                target_dtype = self.bfloat16_dtype
            else:
                target_dtype = "float32"

            if target_dtype == "float32":
                return contextlib.nullcontext()
            elif target_dtype == "float16":
                return torch.cuda.amp.autocast(dtype=torch.float16)
            elif target_dtype == "bfloat16":
                return torch.cuda.amp.autocast(dtype=torch.bfloat16)
            else:
                raise ValueError(f"Invalid bfloat16_dtype: {target_dtype}")

        return torch.cuda.amp.autocast(dtype=dtype)

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
        ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
        if model_name == "eva_clip_g":
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
        #         elif model_name == "eva2_clip_L":
        #             visual_encoder = create_eva2_vit_L(
        #                 img_size, drop_path_rate, use_grad_checkpoint, precision
        #             )
        elif model_name == "clip_L":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        if len(msg.missing_keys) > 0:
            logging.info(f"Missing keys: {len(msg.missing_keys)}")
            self_params = dict(self.named_parameters())
            non_existing_params = [k for k in msg.missing_keys if k not in self_params]
            logging.info(
                f"Non-existing but missing parameters (probably tied weights): "
                f"{non_existing_params}"
            )

            existing_params = [k for k in msg.missing_keys if k in self_params]
            # missing_params = {k: self_params[k] for k in existing_params}
            # print_statedict_info(missing_params)
            missing_counts = Counter([k.split(".")[0] for k in existing_params])
            logging.info(f"Missing counts: {missing_counts.most_common()}")

        logging.info(f"Done loading checkpoint for {type(self).__name__} from {url_or_filename}")

        return msg

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        if self.vit_name == "eva_clip_g":
            vit_num_layers = self.visual_encoder.get_num_layer()
            lr_scales = list(
                lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2)
            )

            parameter_group_names = {}
            parameter_group_vars = {}

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias"):
                    group_name = "no_decay"
                    this_weight_decay = 0.0
                else:
                    group_name = "decay"
                    this_weight_decay = weight_decay
                if "visual_encoder" in name:
                    layer_id = self.visual_encoder.get_num_layer(
                        name.replace("visual_encoder.", "")
                    )
                    group_name = "vit_layer_%d_%s" % (layer_id, group_name)
                else:
                    layer_id = None

                if group_name not in parameter_group_names:
                    if layer_id is not None:
                        scale = lr_scales[layer_id]
                    else:
                        scale = 1
                    parameter_group_names[group_name] = {
                        "weight_decay": this_weight_decay,
                        "params": [],
                        "lr_scale": scale,
                    }
                    parameter_group_vars[group_name] = {
                        "weight_decay": this_weight_decay,
                        "params": [],
                        "lr_scale": scale,
                    }
                parameter_group_vars[group_name]["params"].append(param)
                parameter_group_names[group_name]["params"].append(name)
            # import json
            # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
            optim_params = list(parameter_group_vars.values())
            return optim_params
        else:
            return super().get_optimizer_params(weight_decay, lr_scale)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


@torch.no_grad()
def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(
        model.device
    )

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(
        model.device
    )

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
