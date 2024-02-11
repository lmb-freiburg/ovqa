import logging
import torch
from pathlib import Path

from ovqa.create_prompt import create_prompt
from ovqa.models.x2vlm.model_generation import XVLMForVQA
from ovqa.models.x2vlm.utils import build_tokenizer
from ovqa.models.x2vlm.xvlm import XVLMBase
from ovqa.outputs import QAOutput
from ovqa.common.lavis.registry import registry
from ovqa.models.lavis.base_model import BaseModel
from ovqa.tasks.multimodal_classification import (
    MultimodalClassificationTask,
    MultimodalClassificationSynonymsTask,
    get_classnames_for_classifier,
    get_classifier_cache_file,
    get_classnames_templates_for_classifier,
)


@registry.register_model("x2vlm")
class X2vlmLavis(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "x2vlm_base1b_ftvqa": "ovqa/configs/models/x2vlm/x2vlm_base1b_ftvqa.yaml",
        "x2vlm_base1b_pt": "ovqa/configs/models/x2vlm/x2vlm_base1b_pt.yaml",
        "x2vlm_base1b_pt_itr_imgs384": "ovqa/configs/models/x2vlm/x2vlm_base1b_pt_itr_imgs384.yaml",
        "x2vlm_large1b_ftvqa": "ovqa/configs/models/x2vlm/x2vlm_large1b_ftvqa.yaml",
        "x2vlm_large1b_pt": "ovqa/configs/models/x2vlm/x2vlm_large1b_pt.yaml",
        "x2vlm_large1b_pt_itr_imgs384": "ovqa/configs/models/x2vlm/x2vlm_large1b_pt_itr_imgs384.yaml",
    }

    def load_from_pretrained(self, url_or_filename):
        pass  # model was already loaded by the code before

    def __init__(self, cfg):
        super().__init__()

        # load x2vlm model
        model_name = cfg.model_type
        tokenizer = build_tokenizer(cfg.get("text_encoder", "bert-base-uncased"))
        cfg["pad_token_id"] = tokenizer.pad_token_id
        cfg["eos"] = tokenizer.eos_token
        if model_name.endswith("_ftvqa"):
            model = XVLMForVQA(config=cfg)
        elif model_name.endswith("_pt") or cfg.model_type.endswith("_pt_itr_imgs384"):
            model = XVLMBase(
                config=cfg,
                load_vision_params=False,
                load_text_params=False,
                use_contrastive_loss=True,
                use_matching_loss=False,
                use_mlm_loss=False,
                use_bbox_loss=False,
            )
        else:
            raise ValueError(f"Unknown model name {model_name} for x2vlm.")
        model.tokenizer = tokenizer
        model.load_pretrained(cfg.checkpoint, cfg, is_eval=True)

        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.max_txt_len = cfg.get("max_txt_len", 40)
        self.printed_a_batch = False

    @classmethod
    def from_config(cls, cfg):
        cfg["image_res"] = cfg.get("image_size", 224)
        model = cls(cfg)
        model.load_checkpoint_from_config(cfg)
        return model

    @torch.no_grad()
    def before_evaluation(self, dataset, task_type, **kwargs):
        if task_type not in {MultimodalClassificationTask, MultimodalClassificationSynonymsTask}:
            return
        # get classes for classification
        classnames = get_classnames_for_classifier(dataset, task_type)
        self.classifier_classnames = classnames
        templates = get_classnames_templates_for_classifier(dataset)

        cache_file = get_classifier_cache_file(classnames, templates, self)
        if cache_file.is_file():
            logging.info(f"Load classifier weights from {cache_file}")
            self.classifier = torch.load(cache_file, map_location=self.device)
            return
        logging.info(f"Create zeroshot classifier for {len(classnames)} classes")

        # compute the embeddings of the classes
        zeroshot_weights = []
        for classname in classnames:
            text_classes = [template(classname) for template in templates]  # format with class
            text_input = self.tokenizer(
                text_classes,
                padding="max_length",
                truncation=True,
                max_length=self.cfg["max_tokens"],
                return_tensors="pt",
            ).to(self.device)
            text_embeds = self.model.get_text_embeds(
                text_input.input_ids, text_input.attention_mask
            )
            class_embedding = self.model.get_features(text_embeds=text_embeds).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        self.classifier = torch.stack(zeroshot_weights, dim=1).to(self.device)
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.classifier, cache_file)

    def forward(self, samples):
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, samples, return_embedding=False):
        """Zero-shot classification"""
        image = samples["image"]
        targets = samples["label"]

        # get vision embeddings
        image_embeds, _ = self.model.get_vision_embeds(image)
        image_feat = self.model.get_features(image_embeds=image_embeds)
        # import ipdb; ipdb.set_trace()
        logits = torch.matmul(image_feat, self.classifier) / self.model.temp
        output = {"predictions": logits, "targets": targets}
        if return_embedding:
            output["embeddings"] = image_feat
        return output

    def predict_answers(self, samples, **kwargs):
        return self.generate(samples, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,  # 5,
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
        device = image.device
        batch_size = image.shape[0]

        # get vision embeddings
        image_embeds, image_atts = self.model.get_vision_embeds(image)

        # create question prompt
        text_input = create_prompt(samples, prompt, batch_size)

        # get text input tokens
        question = self.tokenizer(
            text_input,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        # embed image and question in the text_encoder (cross-attention)
        question_output = self.model.text_encoder(
            question.input_ids,
            attention_mask=question.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        question_states = question_output.last_hidden_state
        num_ques = question_states.size(0)
        start_ids = question.input_ids[0, 0].repeat(num_ques, 1)  # bos token

        # generate answer kwargs
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

        # generate answers
        decoder_output = self.model.text_decoder.generate(
            start_ids,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question.attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            **generation_kwargs,
        )

        output_text = self.tokenizer.batch_decode(decoder_output, skip_special_tokens=True)

        if not self.printed_a_batch:
            logging.info(f"Prompt: '{text_input[0]}'")
            logging.info(f"Results of first batch: {output_text[:8]}")
            self.printed_a_batch = True

        if return_dict:
            return QAOutput(answer=output_text)
        else:
            return output_text
