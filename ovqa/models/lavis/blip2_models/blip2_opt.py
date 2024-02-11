"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
import transformers
from packaging import version

# from ovqa.models.lavis.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer, OPTForCausalLM

from ovqa.create_prompt import create_prompt
from ovqa.outputs import QAOutput
from ovqa.common.lavis.registry import registry
from ovqa.models.lavis.blip2_models.blip2 import Blip2Base, disabled_train


@registry.register_model("blip2_opt")
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from ovqa.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "ovqa/configs/lavis_models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "ovqa/configs/lavis_models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "ovqa/configs/lavis_models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "ovqa/configs/lavis_models/blip2/blip2_caption_opt6.7b.yaml",
        "pretrain_optsmalldebug": "ovqa/configs/lavis_models/blip2/blip2_pretrain_optsmalldebug.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        vit_model_num_features=None,  # 1408 for eva-vit-g
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse(
            "4.27"
        ), "BLIP-2 OPT requires transformers>=4.27"

        self.tokenizer = self.init_tokenizer()

        if vit_model != "none":
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
            )
            if freeze_vit:
                for name, param in self.visual_encoder.named_parameters():
                    param.requires_grad = False
                self.visual_encoder = self.visual_encoder.eval()
                self.visual_encoder.train = disabled_train
                logging.info("Done freezing vision encoder")
            vit_model_num_features = self.visual_encoder.num_features
            logging.info(f"vit_model_num_features: {vit_model_num_features}")
        else:
            assert (
                vit_model_num_features is not None
            ), f"No vision model, so vit_model_num_features must be given to the config."

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, vit_model_num_features)

        # delete everything except the query embeddings from QFormer
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # # note: in case you want to input a question to the qformer in order to make the queries
        # # more relevant to the text, disable the deletion above, and enable below.
        # # blip2 does this when finetuning VQA
        # self.Qformer.resize_token_embeddings(len(self.tokenizer))
        # state_dict = self.Qformer.state_dict()
        # for name, param in self.Qformer.named_parameters():
        #     if "_query" in name:
        #         key_orig = name.replace("_query", "")
        #         param.data.copy_(state_dict[key_orig])

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16)
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer("\n", add_special_tokens=False).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        self.printed_a_batch = False

    def forward(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_new_tokens=30,
        min_new_tokens=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        prompt="",
        return_dict=False,
        **kwargs,
    ):
        """

        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use greedy sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_new_tokens (int): The maximum length of the sequence to be generated.
            min_new_tokens (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            length_penalty:
            num_captions (int): Number of captions to be generated for each image.
            temperature:
            prompt (str): The prompt to be prepended to the generated text.
            return_dict:

        Returns:
            A list of strings of length batch_size * num_captions.
            or QAOutput object if return_dict is True.

        """
        image = samples["image"]
        batch_size = image.size(0)
        device = image.device
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(device)

            text_input = create_prompt(samples, prompt, batch_size)
            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(device)

            input_ids = opt_tokens.input_ids
            input_ids_atn = opt_tokens.attention_mask
            attention_mask = torch.cat([atts_opt, input_ids_atn], dim=1)

            generation_kwargs = dict(
                do_sample=use_nucleus_sampling,  # False
                top_p=top_p,  # 0.9
                temperature=temperature,  # 1
                num_beams=num_beams,  # 5
                max_new_tokens=max_new_tokens,  # 30
                min_new_tokens=min_new_tokens,  # 4
                eos_token_id=self.eos_token_id,  # 50118
                repetition_penalty=repetition_penalty,  # 1.0
                length_penalty=length_penalty,  # 1.0
                num_return_sequences=num_captions,  # 1
            )
            self.emit_debug_message(
                text_input, input_ids, self.opt_tokenizer, generation_kwargs, kwargs
            )

            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]

        if not self.printed_a_batch:
            logging.info(f"Results of first batch: {output_text[:8]}")
            raw_output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=False)
            logging.info(f"Raw output of first batch: {raw_output_text[:8]}")

        self.printed_a_batch = True

        if return_dict:
            return QAOutput(answer=output_text)
        else:
            return output_text

    @torch.no_grad()
    def predict_answers(
        self,
        samples,
        **kwargs,
    ):
        return self.generate(samples, **kwargs)

    @torch.no_grad()
    def predict_multiquestion_answers(
        self,
        samples,
        num_beams=5,
        max_new_tokens=10,
        min_new_tokens=1,
        prompt="",
        length_penalty=-1,
        return_dict=False,
        use_nucleus_sampling=False,
        top_p=0.9,
        repetition_penalty=1.0,
        num_captions=1,
        temperature=1,
        inference_method="generate",  # noqa unused
        answer_list=None,  # noqa unused
        num_ans_candidates=128,  # noqa unused
        **kwargs,
    ):
        image = samples["image"]
        batch_size = image.size(0)
        with self.maybe_autocast():
            # process the image
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)

            # process the text
            text_input = create_prompt(samples, prompt, batch_size)
            assert len(text_input) == len(
                image_embeds
            ), f"Batch has different number {len(image_embeds)} of images and sets of questions {len(text_input)}"

            # flatten questions and repeat image embeddings
            all_text_input = []
            len_of_questions = []
            all_inputs_opt = []
            for idx_im, questions in enumerate(text_input):
                n_questions = len(questions)
                len_of_questions.append(n_questions)
                all_text_input.extend(questions)
                idx_inputs_opt = inputs_opt[idx_im]
                all_inputs_opt.append(torch.stack([idx_inputs_opt] * n_questions, axis=0))
            all_inputs_opt = torch.cat(all_inputs_opt, dim=0)

            # tokenize text
            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                all_text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            input_ids = opt_tokens.input_ids
            input_ids_atn = opt_tokens.attention_mask

            generation_kwargs = dict(
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            self.emit_debug_message(
                text_input, input_ids, self.opt_tokenizer, generation_kwargs, kwargs
            )

            prompt_len = input_ids.size(1)

            # divide the image-questions in sets same size of mini batch
            all_answers = []
            for start_idx in range(0, len(all_text_input), batch_size):
                inputs_mini_batch_opt = all_inputs_opt[
                    start_idx : min(start_idx + batch_size, len(all_text_input))
                ]
                atts_opt = torch.ones(inputs_mini_batch_opt.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                text_mini_batch_input_ids = input_ids[
                    start_idx : min(start_idx + batch_size, len(all_text_input))
                ]
                text_mini_batch_input_ids_atn = input_ids_atn[
                    start_idx : min(start_idx + batch_size, len(all_text_input))
                ]

                attention_mask = torch.cat([atts_opt, text_mini_batch_input_ids_atn], dim=1)

                # require transformers>=4.27
                inputs_embeds = self.opt_model.get_input_embeddings()(text_mini_batch_input_ids)
                inputs_embeds = torch.cat([inputs_mini_batch_opt, inputs_embeds], dim=1)

                outputs = self.opt_model.generate(
                    inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generation_kwargs
                )
                output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                output_text = [text.strip() for text in output_text]
                all_answers.extend(output_text)

        if not self.printed_a_batch:
            logging.info(f"Prompt: '{text_input[0]}'")
            logging.info(f"Results of first batch: {all_answers[:8]}")
        self.printed_a_batch = True

        # return list of answers for every image
        idx_answer = 0
        split_answers = []
        for pos_add in len_of_questions:
            split_answers.append(all_answers[idx_answer : idx_answer + pos_add])
            idx_answer += pos_add

        if return_dict:
            return QAOutput(answer=[""] * batch_size, answers=split_answers)
        else:
            return split_answers

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        assert not cfg.get(
            "apply_lemmatizer", False
        ), f"Lemmatizer not supported anymore, should lemmatize after validation is done."
        vit_model_num_features = cfg.get("vit_model_num_features", None)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            vit_model_num_features=vit_model_num_features,
        )
        model.load_checkpoint_from_config(cfg)

        return model
