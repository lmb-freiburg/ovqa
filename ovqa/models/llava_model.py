"""
Model base from lavis / blip2_t5.py
Llava implementation from https://github.com/haotian-liu/LLaVA
"""
import contextlib
import logging
import torch
from attr import define
from copy import deepcopy
from typing import List, Optional

from ovqa.create_prompt import create_prompt
from ovqa.models.llava.conversation import conv_templates, SeparatorStyle, Conversation
from ovqa.models.llava_load import (
    load_llava,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    KeywordsStoppingCriteriaBatched,
)
from ovqa.outputs import QAOutput
from ovqa.common.lavis.dist_utils import is_main_process
from ovqa.common.lavis.registry import registry
from ovqa.models.lavis.base_model import BaseModel


@define
class NoneConversation:
    messages: Optional[list[list[str]]] = None
    roles: tuple[str, str] = ("USER", "ASSISTANT")
    sep_style: int = SeparatorStyle.TWO
    sep: str = " "
    sep2: str = "</s>"
    offset: int = 0  # length of messages

    def __attrs_post_init__(self):
        self.messages: List[List[str]] = [] if self.messages is None else self.messages

    def get_prompt(self):
        return " ".join([message.strip() for role, message in self.messages]).strip()

    def append_message(self, role, message):
        message = "" if message is None else message
        self.messages.append([role, message])

    def copy(self):
        self_copy = deepcopy(self)
        self_copy.messages = list(self_copy.messages)
        return self_copy


@registry.register_model("llava")
class LlavaLavis(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "llava7b": "ovqa/configs/models/llava7b.yaml",
    }

    def load_from_pretrained(self, url_or_filename):
        pass  # model was already loaded by the code before

    def __init__(self, cfg):
        super().__init__()

        img_size = cfg.get("image_size")  # 224 usually
        assert img_size == 224, "LLaVA only supports image size 224"

        # load llava
        model_name = cfg.llm_model
        logging.info(f"Loading LLaVA model: {model_name}")
        model, tokenizer, image_processor, config_dict = load_llava(
            model_name, num_gpus=cfg.num_gpus, gb_per_gpu=cfg.get("gb_per_gpu", 13)
        )
        del image_processor
        self.model = model
        self.tokenizer = tokenizer
        self.image_token_len = config_dict["image_token_len"]
        self.mm_use_im_start_end = config_dict["mm_use_im_start_end"]

        self.conv_mode = cfg.conv_mode
        self.printed_a_batch = False

    def forward(self, samples):
        raise NotImplementedError

    def predict_answers(self, samples, **kwargs):
        return self.generate(samples, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        samples,
        num_beams=1,
        max_new_tokens=1024,
        min_new_tokens=1,
        prompt="",
        length_penalty=1.0,
        use_nucleus_sampling=True,
        top_p=0.9,
        repetition_penalty=1.0,
        num_captions=1,
        temperature=0.2,
        return_dict=False,
        inference_method="generate",  # noqa unused
        answer_list=None,  # noqa unused
        num_ans_candidates=128,  # noqa unused
    ):
        """

        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list[str]): A list of strings of length batch_size.
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
        device = image.device
        batch_size = image.shape[0]
        # assert batch_size == 1, "LLaVA only supports batch size 1"

        # create question prompt
        text_input = create_prompt(samples, prompt, batch_size)

        # add image input dummy tokens
        input_conv_prompt = []
        for text_input_single in text_input:
            if self.mm_use_im_start_end:
                single_input_text_image = "".join(
                    [
                        text_input_single,
                        "\n",
                        DEFAULT_IM_START_TOKEN,
                        DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len,
                        DEFAULT_IM_END_TOKEN,
                    ]
                )
            else:
                single_input_text_image = "".join(
                    [
                        text_input_single,
                        "\n",
                        DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len,
                    ]
                )

            # create conv prompt with the conversation template
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], single_input_text_image)
            conv.append_message(conv.roles[1], None)
            conv_prompt = conv.get_prompt()
            input_conv_prompt.append(conv_prompt)

        self.tokenizer.padding_side = "left"
        tok = self.tokenizer(input_conv_prompt, return_tensors="pt", padding="longest").to(device)
        input_ids = tok.input_ids
        input_ids_mask = tok.attention_mask

        generation_kwargs = dict(
            do_sample=use_nucleus_sampling,
            temperature=temperature,  # 0.2,
            max_new_tokens=max_new_tokens,  # 1024,
            num_beams=num_beams,
            min_new_tokens=min_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )

        if not self.printed_a_batch and is_main_process():
            logging.info(f"Prompt only: '{text_input[0]}'")
            logging.info(f"Conv mode: '{self.conv_mode}'")
        self.emit_debug_message(input_conv_prompt, input_ids, self.tokenizer, generation_kwargs)

        # image should be already correctly preprocessed at this point
        # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteriaBatched(keywords, self.tokenizer, input_ids)

        image_in = image.half().cuda()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=input_ids_mask,
                images=image_in,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_kwargs,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        output_text = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )
        clean_output_text = []
        for single_output in output_text:
            # for each text only keep output until the SEP signal (e.g. "###" for llava 7b)
            single_output = single_output.strip().split(stop_str)[0].strip()
            clean_output_text.append(single_output)

        if not self.printed_a_batch:
            # llava 13b properly finishes generating and produces pad tokens
            logging.info(f"Results of first batch: {clean_output_text}")
            logging.info(
                f"Image stats: {image_in.min()}, {image_in.max()}, "
                f"{image_in.mean()} {image_in.std()}"
            )
            raw_output_text = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=False
            )
            logging.info(f"Raw output of first batch: {raw_output_text[:8]}")

        self.printed_a_batch = True

        if return_dict:
            return QAOutput(answer=clean_output_text)
        else:
            return clean_output_text

    @classmethod
    def from_config(cls, cfg):
        model = cls(cfg)
        model.load_checkpoint_from_config(cfg)
        return model

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16

        if dtype == torch.bfloat16 and hasattr(self, "use_bfloat16") and not self.use_bfloat16:
            # bfloat16 requested, but deactivated via config: no autocast
            return contextlib.nullcontext()

        if self.device == torch.device("cpu"):
            # cpu: no autocast
            return contextlib.nullcontext()

        return torch.cuda.amp.autocast(dtype=dtype)

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        raise NotImplementedError("not needed for inference")


# templates
# NOTE: Its best to just use the original ones
conv_templates["llava_custom_13b"] = Conversation(
    system="You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
    "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "Follow the instructions carefully and answer briefly and concisely.",
    roles=("USER", "ASSISTANT"),
    version="custom",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_templates["llava_custom_7b"] = Conversation(
    system="You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
    "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "Follow the instructions carefully and answer briefly and concisely.",
    roles=("Human", "Assistant"),
    messages=(("Human", "Hi!"), ("Assistant", "Hi there!  How can I help you today?\n")),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
conv_templates["none_13b"] = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_templates["none_7b"] = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=(("Human", "Hi!"), ("Assistant", "Hi there! How can I help you today?")),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
