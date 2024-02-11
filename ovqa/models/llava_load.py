import requests
import torch
from PIL import Image
from io import BytesIO
from transformers import StoppingCriteria, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor

from ovqa.models.llava.model import LlavaLlamaForCausalLM, LlavaMPTForCausalLM
from ovqa.models.llava.utils import disable_torch_init

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len :], skip_special_tokens=True
            )[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


class KeywordsStoppingCriteriaBatched(StoppingCriteria):
    """Fix so it works with batchsize > 1"""

    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            try:
                outputs = self.tokenizer.batch_decode(
                    output_ids[:, self.start_len :], skip_special_tokens=True
                )
            except OverflowError as e:
                # after fixing pad_token_id this should not happen anymore
                # - problem was pad_token_id was -1 in greedy_search
                # but should be tokenizer.pad_token_id
                # - solution was to give pad_token_id=tokenizer.pad_token_id
                # to huggingface generate() method
                raise e
                # overflow_index_tuple = torch.where(output_ids < 0)
                # print(
                #     f"Detected overflow tokens {output_ids[overflow_index_tuple]} at "
                #     f"index {overflow_index_tuple}. Replacing with unk. Original error: "
                #     f"{format_exception(e)}"
                # )

            n_outputs = len(outputs)
            n_keywords_found = 0
            for output in outputs:
                for keyword in self.keywords:
                    if keyword in output:
                        n_keywords_found += 1
                        break
            if n_keywords_found == n_outputs:
                return True
        return False


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_llava(model_name: str, num_gpus: int = 1, gb_per_gpu=13):
    # Model
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Load model {model_name} with {num_gpus} GPUs")

    if num_gpus == 1:
        kwargs = {}
    else:
        kwargs = {
            "device_map": "auto",
            "max_memory": {i: f"{gb_per_gpu}GiB" for i in range(num_gpus)},
        }

    if "mpt" in model_name.lower():
        model = LlavaMPTForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_cache=True,
            **kwargs,
        )
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_cache=True,
            **kwargs,
        )
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower, torch_dtype=torch.float16
    )
    # model.config.mm_vision_tower = 'openai/clip-vit-large-patch14'
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == "meta":
        # noinspection PyProtectedMember
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device="cuda", dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        (
            vision_config.im_start_token,
            vision_config.im_end_token,
        ) = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    if num_gpus == 1:
        model.cuda()

    return (
        model,
        tokenizer,
        image_processor,
        dict(image_token_len=image_token_len, mm_use_im_start_end=mm_use_im_start_end),
    )
