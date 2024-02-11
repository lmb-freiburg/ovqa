from pprint import pprint

from loguru import logger

from ovqa.models.lavis.blip2_models.blip2 import Blip2Base


def check_model_size(model, print_info=True):
    if isinstance(model, Blip2Base):
        keys_dict = {
            "vision": [
                "visual_encoder",
                "ln_vision",
                "vision_proj",
            ],
            "adapter": [
                "Qformer",
                "query_tokens",
                "itm_head",
                "temp",
            ],
            "language": [
                "opt_model",
                "opt_proj",
                "t5_model",
                "t5_proj",
                "llm_model",
                "llm_proj",
                "text_proj",
            ],
        }
    else:
        raise ValueError(f"Unknown model type {type(model)}")

    counts_dict = {k: 0 for k in keys_dict.keys()}
    counts_dict["other"] = 0
    for n, p in model.named_parameters():
        n_key = n.split(".")[0]
        counted = False
        for key_group, keys in keys_dict.items():
            for key in keys:
                if key == n_key:
                    counts_dict[key_group] += p.numel()
                    counted = True
                    break
            if counted:
                break
        if not counted:
            counts_dict["other"] += p.numel()
            logger.info(f"Unknown key {n} with shape {p.shape}")
    if print_info:
        pprint(", ".join(f"{k}={v / 1e9:.3f}B" for k, v in counts_dict.items()))
    return counts_dict


def main():
    from ovqa.models import load_model

    model = load_model("blip2", "pretrain")
    check_model_size(model)


if __name__ == "__main__":
    main()
