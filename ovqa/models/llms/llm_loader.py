import accelerate
import argparse
import os
import torch
from torch.cuda import OutOfMemoryError
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from packg import format_exception
from packg.debugging import connect_to_pycharm_debug_server
from ovqa.models.llms.llm_definitions import MODEL_DICT


def get_device_kwargs(model_name):
    model_info = MODEL_DICT[model_name]
    gpu_settings = model_info["gpu_settings"]
    current_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Detected GPU memory: {current_mem_gb:.1f}G")

    use_settings = None
    for min_mem in sorted(gpu_settings.keys()):
        if current_mem_gb > min_mem:
            use_settings = gpu_settings[min_mem]
        else:
            break
    if use_settings is None:
        raise ValueError(
            f"Could not find GPU settings for {model_name} with "
            f"{current_mem_gb:.1f}G available memory"
        )
    print(f"Found settings for {model_name}: {use_settings}")
    return use_settings


def load_llm(
    model_name: str,
    buffer_size: Optional[int] = None,
    use_auto_gpu_settings: bool = True,
    more_model_kwargs: Optional[dict] = None,
    more_tokenizer_kwargs: Optional[dict] = None,
):
    if use_auto_gpu_settings:
        gpu_settings = get_device_kwargs(model_name)
    else:
        gpu_settings = {}
    more_tokenizer_kwargs = {} if more_tokenizer_kwargs is None else more_tokenizer_kwargs
    more_model_kwargs = {} if more_model_kwargs is None else more_model_kwargs

    model_info = MODEL_DICT[model_name]
    model_name_hf = model_info["name"]

    tokenizer_kwargs = model_info.get("tokenizer_kwargs", {})
    print(f"Load {model_name_hf} tokenizer, kwargs {tokenizer_kwargs} and {more_tokenizer_kwargs}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_hf,
        **tokenizer_kwargs,
        **more_tokenizer_kwargs,
    )

    # ensure pad token and pad left (standard for causallm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # disable progressbar (optional, its annoying in notebooks)
    accelerate.utils.modeling.is_tqdm_available = lambda: False

    model_kwargs = model_info.get("loader_kwargs", {})
    loader_name = model_info.get("loader", "default")
    print(
        f"Load {model_name_hf} loader {loader_name} kwargs: GPU {gpu_settings} "
        f"model {model_kwargs} added {more_model_kwargs}"
    )

    if loader_name == "default":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_hf,
            **gpu_settings,
            **model_kwargs,
            **more_model_kwargs,
        )
    elif loader_name == "gptq":
        from auto_gptq import AutoGPTQForCausalLM, exllama_set_max_input_length  # noqa

        model = AutoGPTQForCausalLM.from_quantized(
            model_name_hf,
            **gpu_settings,
            **model_kwargs,
            **more_model_kwargs,
        )
    elif loader_name == "awq":
        from awq import AutoAWQForCausalLM

        # fix TypeError: from_quantized() got an unexpected keyword argument 'device_map'
        if "device_map" in gpu_settings:
            del gpu_settings["device_map"]

        model = AutoAWQForCausalLM.from_quantized(
            model_name_hf,
            **gpu_settings,
            **model_kwargs,
            **more_model_kwargs,
        )

    else:
        raise ValueError(f"Unknown loader {loader_name}")

    if buffer_size is not None:
        print(f"Adjust buffersize to {buffer_size}")
        try:
            model = exllama_set_max_input_length(model, buffer_size)
        except Exception as e:
            print(f"Failed setting buffer size: {format_exception(e)}")

    model.eval()
    print(f"Done loading model: {type(model)}")
    for try_attr in ["device", "dtype", "hf_device_map"]:
        try:
            print(f"Model attribute {try_attr}: {getattr(model, try_attr)}")
        except AttributeError:
            pass

    return model, tokenizer


def run_llm_batch(model, tokenizer, prompts, device="cuda:0", **kwargs):
    token_dict = tokenizer(prompts, return_tensors="pt", padding="longest").to(device)
    with torch.inference_mode():
        try:
            output_ids = model.generate(**token_dict, **kwargs)
        except OutOfMemoryError as e:
            os.system("nvidia-smi")
            raise e

    output_ids_cut = output_ids[:, token_dict["input_ids"].shape[1] :]
    return output_ids_cut


def testrun_forward(model, tokenizer, device="cuda:0"):
    token_dict = tokenizer(["Hello", "Hello world and"], return_tensors="pt", padding="longest").to(
        device
    )
    with torch.inference_mode():
        try:
            output_dict = model.forward(**token_dict, return_dict=True)
        except OutOfMemoryError as e:
            os.system("nvidia-smi")
            raise e
    print(token_dict.input_ids.shape)
    print(output_dict.logits.shape)
    kvs = output_dict.past_key_values
    # tuple (length layers) of tuple (keys, values)
    print(len(kvs))  # 32 = one for each layer
    kvs_l0 = kvs[0]
    print(len(kvs_l0))  # 2 = keys and values
    print(kvs_l0[0].shape)  # ([2=batch, 32=heads, 4=sequence, 128=hidden_dim_head])
    print(f"Done checking KV cache")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=str, help="Connect debug server on this host.")
    parser.add_argument(
        "--trace_port", type=int, default=33553, help="Target debugging server port"
    )
    # parser.add_argument("-m", "--model", type=str, default="llama2-7b-chat-4bit")
    parser.add_argument("-m", "--model", type=str, default="llama2-7b-chat-4bit")
    args = parser.parse_args()
    if args.trace is not None:
        connect_to_pycharm_debug_server(args.trace, args.trace_port)

    model, tokenizer = load_llm(args.model)
    testrun_forward(model, tokenizer)

    # this prompt failed with llamav2 kernels, so disabled them for now.
    prompts = [
        "Tell me about gravity:",
        "I would like to",
        "Hey my friend",
        "Tell me about gravity",
        "I am",
        "Today I am in Paris and",
        "Tell me about gravity",
    ]
    output_ids_cut = run_llm_batch(
        model, tokenizer, prompts, device=model.device, max_new_tokens=10
    )

    for nb, output_id in enumerate(output_ids_cut):
        print(f"Prompt: {prompts[nb]}")
        print(f"Generated: {tokenizer.decode(output_id, skip_special_tokens=False)}")
        print()


if __name__ == "__main__":
    main()
