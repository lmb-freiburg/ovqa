"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
from typing import List

import torch
import torch.nn as nn
from transformers import T5TokenizerFast

from ovqa.create_prompt import create_prompt
from ovqa.outputs import QAOutput
from ovqa.common.lavis.registry import registry
from ovqa.models.lavis.blip2_models.blip2 import Blip2Base, disabled_train
from ovqa.models.lavis.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration


@registry.register_model("blip2_t5")
class Blip2T5(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xl_vitL: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from ovqa.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "ovqa/configs/lavis_models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xl_vitL": "ovqa/configs/lavis_models/blip2/blip2_pretrain_flant5xl_vitL.yaml",
        "pretrain_flant5xxl": "ovqa/configs/lavis_models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "ovqa/configs/lavis_models/blip2/blip2_caption_flant5xl.yaml",
        "pretrain_flant5smalldebug": "ovqa/configs/lavis_models/blip2/blip2_pretrain_flant5smalldebug.yaml",
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
        t5_model="google/flan-t5-xl",
        max_txt_len=32,
        bfloat16_dtype="bfloat16",
        num_gpus=1,
        gb_per_gpu=13,
        force_autocast=False,
        do_vqa_finetune=False,
        add_q_to_qformer=False,
    ):
        super().__init__()
        self.bfloat16_dtype = bfloat16_dtype
        self.use_bfloat16 = bfloat16_dtype == "bfloat16"
        self.force_autocast = force_autocast
        self.do_vqa_finetune = do_vqa_finetune
        self.add_q_to_qformer = add_q_to_qformer  # set True for eval on finetuned vqa model

        logging.info(
            f"Create {type(self).__name__} with bfloat16_dtype={self.bfloat16_dtype} "
            f"model {t5_model}"
        )
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("Done freezing vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        if not self.do_vqa_finetune and not self.add_q_to_qformer:
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"

        if num_gpus == 1:
            kwargs = {}
        else:
            kwargs = dict(
                device_map="auto",
                max_memory={i: f"{gb_per_gpu}GiB" for i in range(num_gpus)},
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )

        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config, **kwargs
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            if self.use_bfloat16:
                param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)

        self.max_txt_len = max_txt_len

        if num_gpus > 1:
            # in case of multi gpus we have to manually move all the other stuff on gpu 0
            other_device = "cuda:0"
            self.visual_encoder = self.visual_encoder.to(other_device)
            self.ln_vision = self.ln_vision.to(other_device)
            self.Qformer = self.Qformer.to(other_device)
            # self.query_tokens = self.query_tokens.to(other_device)
            self.t5_proj = self.t5_proj.to(other_device)
        self.num_gpus = num_gpus

    def forward_vqa_finetuning(self, samples):
        """

        Args:
            samples: dict of
                image: tensor (B, 3, 490, 490) float32
                text_input: list[str] length B, already formatted questions
                answer: list[str] length N_answers
                weight: tensor (N_answers,) float32 weights for each answer
                n_answers: tensor (B,) int64 to map answers to questions
                prompt: str: prompt including "{}" to format the questions
                epoch: int
                num_iters_per_epoch: int
                iters: int
        Returns:

        """
        # finetuning vqa mode
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        device = image_embeds.device
        batch_size: int = image_embeds.shape[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        prompt = samples["prompt"]
        placeholder = "{}"
        assert placeholder in prompt, (
            f"Require prompt with a " f"{placeholder} placeholder but got prompt '{prompt}'"
        )
        questions_without_prompt: List[str] = samples["text_input"]
        questions_with_prompt = [prompt.format(question) for question in questions_without_prompt]
        questions_tokens = self.tokenizer(
            questions_without_prompt,
            padding="max_length",  # for safety just use this, shouldnt make much diff
            # padding="longest",  # lets see if this works, should be faster
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        questions_input_ids = questions_tokens.input_ids
        questions_attention_mask = questions_tokens.attention_mask
        query_atts = torch.ones(query_tokens.shape[:-1], dtype=torch.long).to(device)
        attention_mask_all = torch.cat([query_atts, questions_attention_mask], dim=1)
        query_output = self.Qformer.bert(
            questions_input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        # input to qformer is 32 query embeddings + the word embeddings of the question
        # from the output, take only the first 32 query embeddings and discard
        # the output text embeddings
        qformer_last_hidden_state = query_output.last_hidden_state[:, : query_tokens.shape[1]]

        # linear project to t5 input size
        inputs_t5 = self.t5_proj(qformer_last_hidden_state)  # (B, 32, 2048)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)  # (B, 32)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                questions_with_prompt,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            input_tokens_ids = input_tokens.input_ids  # (B, seq_len)
            input_tokens_mask = input_tokens.attention_mask

            labels_text = samples["answer"]
            output_tokens = self.t5_tokenizer(
                labels_text,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(
                device
            )  # (N_answers, seq_len)
            output_tokens_ids = output_tokens.input_ids
            output_tokens_mask = output_tokens.attention_mask

            encoder_atts = torch.cat([atts_t5, input_tokens_mask], dim=1)  # (B, 32 + seq_len)

            targets = output_tokens_ids.masked_fill(
                output_tokens_ids == self.t5_tokenizer.pad_token_id, -100
            )  # (N_answers, seq_len)

            inputs_embeds_text = self.t5_model.encoder.embed_tokens(
                input_tokens_ids
            )  # (B, seq_len, H)
            inputs_embeds_all = torch.cat(
                [inputs_t5, inputs_embeds_text], dim=1
            )  # (B, 32 + seq_len, H)

            n_answers = samples["n_answers"]  # (B,) int64 number of answers per batch
            inputs_embeds_repeated = torch.repeat_interleave(
                inputs_embeds_all, n_answers, dim=0
            )  # (N_answers, 32 + seq_len, H)
            encoder_atts_repeated = torch.repeat_interleave(
                encoder_atts, n_answers, dim=0
            )  # (N_answers, 32 +seq_len)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds_repeated,
                attention_mask=encoder_atts_repeated,
                decoder_attention_mask=output_tokens_mask,
                return_dict=True,
                labels=targets,
                reduction="none",
            )
            loss_full = outputs.loss  # shape (N_answers,)
            loss_weights = samples["weight"]  # shape (N_answers,)
            loss = (loss_full * loss_weights / batch_size).sum()
            return {"loss": loss}

    def forward(self, samples):
        if self.do_vqa_finetune:
            return self.forward_vqa_finetuning(samples)

        # pretraining stage 2 mode
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        device = image_embeds.device
        batch_size: int = image_embeds.shape[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        qformer_last_hidden_state = query_output.last_hidden_state

        inputs_t5 = self.t5_proj(qformer_last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            output_tokens = self.t5_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
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
        inference_method="generate",  # noqa unused
        answer_list=None,  # noqa unused
        num_ans_candidates=128,  # noqa unused
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
        batch_size = image.shape[0]
        device = image.device
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_tokens = query_tokens.to(device)

        if self.add_q_to_qformer:
            # for vqa finetuned model we need to add the question without prompt to qformer
            # but then take only the query outputs. to make querys more relevant to question.
            questions_without_prompt = samples["text_input"]
            questions_tokens = self.tokenizer(
                questions_without_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            questions_input_ids = questions_tokens.input_ids
            questions_attention_mask = questions_tokens.attention_mask
            query_atts = torch.ones(query_tokens.shape[:-1], dtype=torch.long).to(device)
            attention_mask_all = torch.cat([query_atts, questions_attention_mask], dim=1)
            query_output = self.Qformer.bert(
                questions_input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask_all,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            qformer_last_hidden_state = query_output.last_hidden_state[:, : query_tokens.shape[1]]
        else:
            # normal qformer usage
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            qformer_last_hidden_state = query_output.last_hidden_state

        inputs_t5 = self.t5_proj(qformer_last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)

        text_input = create_prompt(samples, prompt, batch_size)
        input_tokens = self.t5_tokenizer(text_input, padding="longest", return_tensors="pt").to(
            device
        )

        # do not cut the end of sentence token </s> from the prompt since that will be input
        # to the t5 encoder
        input_ids = input_tokens.input_ids
        input_ids_atn = input_tokens.attention_mask

        encoder_atts = torch.cat([atts_t5, input_ids_atn], dim=1)

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
        self.emit_debug_message(text_input, input_ids, self.t5_tokenizer, generation_kwargs, kwargs)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            # at this point inputs_embeds contains the image and prompt embeddings
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds, attention_mask=encoder_atts, **generation_kwargs
            )
            output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if not self.printed_a_batch:
            logging.info(f"Prompt: '{text_input[0]}'")
            logging.info(f"Results of first batch: {output_text[:8]}")
            self.printed_a_batch = True

        if return_dict:
            return QAOutput(answer=output_text)
        else:
            return output_text

    def predict_answers(
        self,
        samples,
        **kwargs,
    ):
        return self.generate(samples, **kwargs)

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
        batch_size = image.shape[0]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if self.add_q_to_qformer:
            raise NotImplementedError(f"Multiquestion and {self.add_q_to_qformer=} not implemented")
            # for vqa finetuned model we need to add the question without prompt to qformer
            # but then take only the query outputs. to make querys more relevant to question.
            # see generate() for how to do this
            # 1) get questions without prompts and flatten them, tokenize them
            # 2) repeat image embeds for each question
            # 3) run qformer with question
            # 4) take only query embeds (first query_tokens.shape[1] = 32)
            # now you have 1 query embed for each of the flat multi question
            # proceed similar to below and create all_inputs_t5

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        # process the text
        text_input = create_prompt(samples, prompt, batch_size)
        assert len(text_input) == len(
            image_embeds
        ), f"Batch has different number {len(image_embeds)} of images and sets of questions {len(text_input)}"

        # flatten questions and repeat image embeddings
        all_text_input = []
        len_of_questions = []
        all_inputs_t5 = []
        for idx_im, questions in enumerate(text_input):
            n_questions = len(questions)
            len_of_questions.append(n_questions)
            all_text_input.extend(questions)
            idx_inputs_t5 = inputs_t5[idx_im]
            all_inputs_t5.append(torch.stack([idx_inputs_t5] * n_questions, dim=0))
        all_inputs_t5 = torch.cat(all_inputs_t5, dim=0)

        # tokenize text
        input_tokens = self.t5_tokenizer(all_text_input, padding="longest", return_tensors="pt").to(
            image.device
        )

        # do not cut the end of sentence token </s> from the prompt since that will be input
        # to the t5 encoder
        input_ids = input_tokens.input_ids
        input_ids_atn = input_tokens.attention_mask

        # encoder_atts = torch.cat([atts_t5, input_ids_atn], dim=1)

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
        self.emit_debug_message(text_input, input_ids, self.t5_tokenizer, generation_kwargs, kwargs)

        prompt_len = input_ids.size(1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            # divide the image-questions in sets same size of mini batch
            all_answers = []
            for start_idx in range(0, len(all_text_input), batch_size):
                inputs_mini_batch_t5 = all_inputs_t5[
                    start_idx : min(start_idx + batch_size, len(all_text_input))
                ]
                atts_t5 = torch.ones(inputs_mini_batch_t5.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                text_mini_batch_input_ids = input_ids[
                    start_idx : min(start_idx + batch_size, len(all_text_input))
                ]
                text_mini_batch_input_ids_atn = input_ids_atn[
                    start_idx : min(start_idx + batch_size, len(all_text_input))
                ]

                encoder_atts = torch.cat([atts_t5, text_mini_batch_input_ids_atn], dim=1)

                inputs_embeds = self.t5_model.encoder.embed_tokens(text_mini_batch_input_ids)
                inputs_embeds = torch.cat([inputs_mini_batch_t5, inputs_embeds], dim=1)

                # at this point inputs_embeds contains the image and prompt embeddings
                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds, attention_mask=encoder_atts, **generation_kwargs
                )
                output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                output_text = [text.strip() for text in output_text]

                all_answers.extend(output_text)

        if not self.printed_a_batch:
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
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")
        max_txt_len = cfg.get("max_txt_len", 32)
        assert not cfg.get(
            "apply_lemmatizer", False
        ), f"Lemmatizer not supported anymore, should lemmatize after validation is done."
        bfloat16_dtype = cfg.get("bfloat16_dtype", "bfloat16")
        num_gpus = cfg.get("num_gpus", 1)
        gb_per_gpu = cfg.get("gb_per_gpu", 13)
        force_autocast = cfg.get("force_autocast", False)
        do_vqa_finetune = cfg.get("do_vqa_finetune", False)
        add_q_to_qformer = cfg.get("add_q_to_qformer", False)
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            max_txt_len=max_txt_len,
            bfloat16_dtype=bfloat16_dtype,
            num_gpus=num_gpus,
            gb_per_gpu=gb_per_gpu,
            force_autocast=force_autocast,
            do_vqa_finetune=do_vqa_finetune,
            add_q_to_qformer=add_q_to_qformer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
