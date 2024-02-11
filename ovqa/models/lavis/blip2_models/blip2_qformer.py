"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from ovqa.create_prompt import create_prompt
from ovqa.outputs import QAOutput
from ovqa.common.lavis import dist_utils
from ovqa.common.lavis.dist_utils import is_main_process
from ovqa.common.lavis.registry import registry
from ovqa.models.lavis.base_model import all_gather_with_grad, concat_all_gather
from ovqa.models.lavis.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from ovqa.models.lavis.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

# for retrival
from ovqa.tasks.multimodal_classification import (
    MultimodalClassificationTask,
    MultimodalClassificationSynonymsTask,
    get_classifier_cache_file,
    get_classnames_for_classifier,
    get_classnames_templates_for_classifier,
)


@registry.register_model("blip2")
@registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from ovqa.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "ovqa/configs/lavis_models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "ovqa/configs/lavis_models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "ovqa/configs/lavis_models/blip2/blip2_coco.yaml",
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
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        vit_model_num_features=None,  # 1408 for eva-vit-g
        cut_sep=False,
        replace_bos=False,
        use_itm=False,
        k_test=128,
    ):
        super().__init__()

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
                logging.info("Done freezing vision encoder.")
            vit_model_num_features = self.visual_encoder.num_features
            logging.info(f"vit_model_num_features: {vit_model_num_features}")
        else:
            assert (
                vit_model_num_features is not None
            ), f"No vision model, so vit_model_num_features must be given to the config."

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, vit_model_num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.cut_sep = cut_sep
        self.replace_bos = replace_bos
        self.use_itm = use_itm
        self.k_test = k_test

        self.printed_a_batch = False

    def forward(self, samples):
        image = samples["image"]
        text = samples["text_input"]

        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        # rank = dist.get_rank()
        rank = dist_utils.get_rank()
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(image.device)

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(image.device)
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(image.device)

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_new_tokens=30,
        min_new_tokens=10,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        num_captions=1,
        temperature=1,
        prompt="",
        return_dict=False,
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
        with self.maybe_autocast():
            if "image" in samples:
                image = samples["image"]
                image_embeds = self.ln_vision(self.visual_encoder(image))
                device = image.device
                batch_size = image.size(0)
            else:
                image_embeds = samples["vit_feat"].to(self.device)
                device = image_embeds.device
                batch_size = image_embeds.size(0)

        image_embeds = image_embeds.float()
        if use_nucleus_sampling:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        text_input = create_prompt(samples, prompt, batch_size)
        input_ids = self.tokenizer(
            text_input, return_tensors="pt", max_length=30, truncation=True
        ).input_ids.to(device)

        # setting below flags to true sometimes breaks the model
        if self.cut_sep:
            input_ids = input_ids[:, :-1]  # cut off SEP token
        if self.replace_bos:
            input_ids[:, 0] = self.tokenizer.bos_token_id  # set BOS as first token

        if not self.printed_a_batch and is_main_process():
            logging.info(f"Prompt: '{text_input[0]}'")
            logging.info(f"Input ids: {input_ids.shape} " f"0: {input_ids[0].tolist()}")
            logging.info(
                f"Decoded prompt: "
                f"{self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}"
            )

        prompt_len = input_ids.size(1)
        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_captions,
            **model_kwargs,
        )
        outputs = outputs[:, prompt_len:]
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if not self.printed_a_batch:
            logging.info(f"Results of first batch: {captions[:8]}")
        self.printed_a_batch = True

        if return_dict:
            return QAOutput(answer=captions)
        else:
            return captions

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_new_tokens=10,
        min_new_tokens=1,
        prompt="",
        length_penalty=-1,
        return_dict=False,
        **kwargs,
    ):
        assert (
            inference_method == "generate"
        ), f"Only support generate method, got {inference_method}"
        with self.maybe_autocast():
            if "image" in samples:
                image = samples["image"]
                image_embeds = self.ln_vision(self.visual_encoder(image))
                device = image.device
                batch_size = image.size(0)
            else:
                image_embeds = samples["vit_feat"].to(self.device)
                device = image_embeds.device
                batch_size = image_embeds.size(0)
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        text_input = create_prompt(samples, prompt, batch_size)

        batch_encoding = self.tokenizer(text_input, padding="longest", return_tensors="pt").to(
            device
        )
        input_ids = batch_encoding.input_ids

        # setting below flags to true sometimes breaks the model
        if self.cut_sep:
            input_ids = input_ids[:, :-1]  # cut off SEP token
        if self.replace_bos:
            input_ids[:, 0] = self.tokenizer.bos_token_id  # set BOS as first token

        if not self.printed_a_batch and is_main_process():
            logging.info(f"Prompt: '{text_input[0]}'")
            logging.info(f"Input ids: {input_ids.shape} " f"0: {input_ids[0].tolist()}")
            logging.info(
                f"Decoded prompt: "
                f"{self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}"
            )

        prompt_len = input_ids.size(1)
        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            # do_sample=use_nucleus_sampling,
            # top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            length_penalty=length_penalty,
            **model_kwargs,
        )
        outputs = outputs[:, prompt_len:]
        answer = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if not self.printed_a_batch:
            logging.info(f"Results of first batch: {answer[:8]}")
        self.printed_a_batch = True

        if return_dict:
            return QAOutput(answer=answer)
        else:
            return answer

    def predict_multiquestion_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_new_tokens=10,
        min_new_tokens=1,
        prompt="",
        length_penalty=-1,
        return_dict=False,
        **kwargs,
    ):
        assert (
            inference_method == "generate"
        ), f"Only support generate method, got {inference_method}"
        with self.maybe_autocast():
            if "image" in samples:
                image = samples["image"]
                image_embeds = self.ln_vision(self.visual_encoder(image))
                device = image.device
                batch_size = image.size(0)
            else:
                image_embeds = samples["vit_feat"].to(self.device)
                device = image_embeds.device
                batch_size = image_embeds.size(0)
        image_embeds = image_embeds.float()

        # process the text
        text_input = create_prompt(samples, prompt, batch_size)
        assert len(text_input) == len(
            image_embeds
        ), f"Batch has different number {len(image_embeds)} of images and sets of questions {len(text_input)}"

        # flatten questions and repeat image embeddings
        all_text_input = []
        len_of_questions = []
        all_image_embeds = []
        for idx_im, questions in enumerate(text_input):
            n_questions = len(questions)
            len_of_questions.append(n_questions)
            all_text_input.extend(questions)
            idx_im_embeds = image_embeds[idx_im]
            all_image_embeds.append(torch.stack([idx_im_embeds] * n_questions, axis=0))
        all_image_embeds = torch.cat(all_image_embeds, dim=0)

        batch_encoding = self.tokenizer(all_text_input, padding="longest", return_tensors="pt").to(
            device
        )
        input_ids = batch_encoding.input_ids

        # setting below flags to true sometimes breaks the model
        if self.cut_sep:
            input_ids = input_ids[:, :-1]  # cut off SEP token
        if self.replace_bos:
            input_ids[:, 0] = self.tokenizer.bos_token_id  # set BOS as first token

        if not self.printed_a_batch and is_main_process():
            logging.info(f"Prompt: '{all_text_input[:8]}'")
            logging.info(f"Input ids: {input_ids.shape} " f"0: {input_ids[:8].tolist()}")
            logging.info(
                f"Decoded prompt: "
                f"{self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}"
            )

        prompt_len = input_ids.size(1)

        # divide the image-questions in sets same size of mini batch
        all_answers = []
        for start_idx in range(0, len(all_text_input), batch_size):
            image_mini_batch_embeds = all_image_embeds[
                start_idx : min(start_idx + batch_size, len(all_text_input))
            ]
            text_mini_batch_input_ids = input_ids[
                start_idx : min(start_idx + batch_size, len(all_text_input))
            ]

            # build image input by repeating the image as many number of questions
            image_atts = torch.ones(image_mini_batch_embeds.size()[:-1], dtype=torch.long).to(
                device
            )
            model_kwargs = {
                "encoder_hidden_states": image_mini_batch_embeds,
                "encoder_attention_mask": image_atts,
            }
            query_tokens = self.query_tokens.expand(image_mini_batch_embeds.shape[0], -1, -1)

            outputs = self.Qformer.generate(
                input_ids=text_mini_batch_input_ids,
                query_embeds=query_tokens,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=num_beams,
                # do_sample=use_nucleus_sampling,
                # top_p=top_p,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                length_penalty=length_penalty,
                **model_kwargs,
            )
            outputs = outputs[:, prompt_len:]
            answer = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_answers.extend(answer)

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

    @torch.no_grad()
    def predict(self, samples):
        """Zero-shot classification"""
        image = samples["image"]
        targets = samples["label"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)
        sim_q2t = torch.matmul(image_feats.unsqueeze(1), self.classifier.unsqueeze(-1)).squeeze()
        # [batch_size, num_classes, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)

        if self.use_itm:
            # try to improve the retrieval logits using the ITM head
            for i, sims in enumerate(sim_i2t):
                topk_sim, topk_idx = sims.topk(k=self.k_test, dim=0)
                image_inputs = image_embeds[i].repeat(self.k_test, 1, 1)
                score = self.compute_itm(
                    image_inputs=image_inputs,
                    text_ids=self.classifier_itm_text_ids[topk_idx],
                    text_atts=self.classifier_itm_text_atts[topk_idx],
                ).float()
                sim_i2t[i, topk_idx] = score + topk_sim

        logits = 100 * sim_i2t

        return {"predictions": logits, "targets": targets}

    @torch.no_grad()
    def before_evaluation(self, dataset, task_type, **kwargs):
        if task_type not in {MultimodalClassificationTask, MultimodalClassificationSynonymsTask}:
            return
        classnames = get_classnames_for_classifier(dataset, task_type)
        self.classifier_classnames = classnames

        if self.use_itm:
            # for ITM, get the classes as tokens
            num_text = len(classnames)
            text_bs = 256
            text_ids, text_atts = [], []
            # try a single prompt to improve it
            classnames_prompted = [f"a photo of a {c}." for c in classnames]
            for i in range(0, num_text, text_bs):
                text = classnames_prompted[i : min(num_text, i + text_bs)]

                text_input = self.tokenizer(
                    text, padding="max_length", truncation=True, max_length=35, return_tensors="pt"
                ).to(self.device)
                text_ids.append(text_input.input_ids)
                text_atts.append(text_input.attention_mask)
            self.classifier_itm_text_ids = torch.cat(text_ids, dim=0)
            self.classifier_itm_text_atts = torch.cat(text_atts, dim=0)

        templates = get_classnames_templates_for_classifier(dataset)
        cache_file = get_classifier_cache_file(classnames, templates, self)
        if cache_file.is_file():
            logging.info(f"Load classifier weights from {cache_file}")
            self.classifier = torch.load(cache_file, map_location=self.device)
            return
        logging.info(f"Create zeroshot classifier for {len(classnames)} classes")

        zeroshot_weights = []
        for classname in classnames:
            text_classes = [template(classname) for template in templates]  # format with class

            text_tokens = self.tokenizer(
                text_classes,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            text_output = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            text_feat = text_output.last_hidden_state[:, 0, :]
            class_embedding = F.normalize(
                self.text_proj(text_feat).mean(dim=0, keepdim=True), dim=-1
            )
            zeroshot_weights.append(class_embedding)

        self.classifier = torch.stack(zeroshot_weights, dim=1).to(self.device)
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.classifier, cache_file)

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(image_inputs.device)
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_inputs.device)
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert image is not None, "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(
                self.device
            )
            query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert caption is not None, "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(self.device)

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(
                self.device
            )
            query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(self.device)
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        vit_model_num_features = cfg.get("vit_model_num_features", None)
        use_itm = cfg.get("use_itm", False)
        k_test = cfg.get("k_test", 128)

        assert not cfg.get(
            "apply_lemmatizer", False
        ), f"Lemmatizer not supported anymore, should lemmatize after validation is done."

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            vit_model_num_features=vit_model_num_features,
            cut_sep=cfg.get("cut_sep", False),
            replace_bos=cfg.get("replace_bos", False),
            use_itm=use_itm,
            k_test=k_test,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
