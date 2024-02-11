"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging

import torch

from ovqa.create_prompt import create_prompt
from ovqa.outputs import QAOutput
from ovqa.common.lavis.registry import registry
from ovqa.models.lavis.blip_models.blip import BlipBase
from ovqa.models.lavis.blip_models.blip_outputs import (
    BlipOutput,
    BlipIntermediateOutput,
)
from ovqa.models.lavis.med import XBertEncoder, XBertLMHeadDecoder
from ovqa.models.lavis.vit import VisionTransformerEncoder


@registry.register_model("blip_vqa")
class BlipVQA(BlipBase):
    """
    BLIP VQA models.

    Supported model types:
        - base: vqa model initialized with pre-trained BLIP base model on 115M image-text pairs after CapFilt; not fine-tuned.
        - vqav2: fine-tuned BLIP base model on VQA v2.0 dataset.

    Usage:
        >>> from ovqa.models import load_model
        >>> model = load_model("blip_vqa", "vqav2")
        >>> model = load_model("blip_vqa", "okvqa")
        >>> model = load_model("blip_vqa", "aokvqa")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vqav2": "ovqa/configs/lavis_models/other/blip_vqav2.yaml",
        "okvqa": "ovqa/configs/lavis_models/other/blip_vqa_okvqa.yaml",
        "aokvqa": "ovqa/configs/lavis_models/other/blip_vqa_aokvqa.yaml",
    }

    def __init__(self, image_encoder, text_encoder, text_decoder, max_txt_len=35):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.max_txt_len = max_txt_len

    def forward(self, samples):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480.
                - text_input (list): A list of strings, each string is a question
                - answer (list): A list of strings, each string is an answer
                - weight (torch.Tensor): A tensor used to weigh each answer in the loss computation.
                   The shape of the tensor is (sum(n_answers),)
                - n_answers (torch.Tensor): A tensor shape (batch_size,) containing the number of answers
                     for each question in the batch.

        Returns:
            A BlipOutput object containing loss and intermediate outputs,
            see :class:`ovqa.models.lavis.blip_outputs.BlipOutput` for more details.

        Examples:
        ```python
            >>> import torch
            >>> from ovqa.models import load_model
            >>> model = load_model("blip_vqa")
            >>> samples = {
            ...     "image": torch.rand(2, 3, 480, 480),
            ...     "text_input": ["What is this?", "What is that?"],
            ...     "answer": ["cat", "cat", "dog"],
            ...     "weight": torch.tensor([1.0, 1.0, 1.0]),
            ...     "n_answers": torch.tensor([2, 1]),
            ... }
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['intermediate_output', 'loss'])
            >>> output.intermediate_output.keys()
            odict_keys(['image_embeds', 'encoder_output', 'decoder_output', 'decoder_labels'])
        ```
        """
        encoder_output, image_embeds = self.forward_encoder(samples)
        loss, decoder_output, decoder_targets = self.forward_decoder(
            samples=samples, encoder_out=encoder_output
        )

        return BlipOutput(
            loss=loss,
            intermediate_output=BlipIntermediateOutput(
                image_embeds=image_embeds,
                encoder_output=encoder_output,
                decoder_output=decoder_output,
                decoder_labels=decoder_targets,
            ),
        )

    def forward_encoder(self, samples):
        questions = samples["text_input"]
        questions = self.tokenizer(
            questions,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        questions.input_ids[:, 0] = self.tokenizer.enc_token_id
        samples.update({"tokenized_text": questions})

        image_embeds = self.visual_encoder.forward_features(samples["image"])
        encoder_output = self.text_encoder.forward_automask(
            tokenized_text=samples["tokenized_text"], visual_embeds=image_embeds
        )

        return encoder_output, image_embeds

    def forward_decoder(self, samples, encoder_out, **kwargs):
        answers = self.tokenizer(samples["answer"], padding="longest", return_tensors="pt").to(
            self.device
        )
        answers.input_ids[:, 0] = self.tokenizer.bos_token_id
        answer_targets = answers.input_ids.masked_fill(
            answers.input_ids == self.tokenizer.pad_token_id, -100
        )

        question_states = []
        question_atts = []

        question = samples["tokenized_text"]
        question_output = encoder_out

        for b, n in enumerate(samples["n_answers"]):
            question_states += [question_output.last_hidden_state[b]] * n
            question_atts += [question.attention_mask[b]] * n

        question_states = torch.stack(question_states, dim=0)
        question_atts = torch.stack(question_atts, dim=0)

        answer_output = self.text_decoder(
            answers.input_ids,
            attention_mask=answers.attention_mask,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=answer_targets,
            return_dict=True,
            reduction="none",
        )

        loss = samples["weight"] * answer_output.loss
        bsz = samples["image"].size(0)

        loss = loss.sum() / bsz

        return loss, answer_output, answer_targets

    def predict_answers(
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
                - text_input (list, Optional): A list of strings, each string is a question
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
        image_embeds = self.visual_encoder.forward_features(image)

        batch_size = image.shape[0]
        device = image.device

        text_input = create_prompt(samples, prompt, batch_size)

        input_tokens = self.tokenizer(
            text_input,
            padding="longest",
            return_tensors="pt",
        ).to(device)
        input_tokens.input_ids[:, 0] = self.tokenizer.enc_token_id

        question_output = self.text_encoder.forward_automask(
            tokenized_text=input_tokens, visual_embeds=image_embeds
        )

        # question_states = question_output.last_hidden_state.repeat_interleave(
        #     num_beams, dim=0
        # )
        question_states = question_output.last_hidden_state
        question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long).to(device)

        model_kwargs = {
            "encoder_hidden_states": question_states,
            "encoder_attention_mask": question_atts,
        }

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
            text_input, input_tokens.input_ids, self.tokenizer, generation_kwargs, kwargs
        )

        bos_ids = torch.full(
            (batch_size, 1), fill_value=self.tokenizer.bos_token_id, device=self.device
        )
        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs,
            **generation_kwargs,
        )

        # collect answers
        answers = []
        for output in outputs:
            answer = self.tokenizer.decode(output, skip_special_tokens=True)
            answers.append(answer)

        if not self.printed_a_batch:
            logging.info(f"Prompt: '{text_input[0]}'")
            logging.info(f"Results of first batch: {answers[:8]}")
            self.printed_a_batch = True

        if return_dict:
            return QAOutput(answer=answers)
        else:
            return answers

    @classmethod
    def from_config(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.from_config(cfg)

        # text encoder + multimodal encoder
        text_encoder = XBertEncoder.from_config(cfg)
        text_decoder = XBertLMHeadDecoder.from_config(cfg)

        max_txt_len = cfg.get("max_txt_len", 35)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_decoder=text_decoder,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)

        return model
