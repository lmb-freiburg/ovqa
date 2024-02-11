from auto_gptq.modeling import LlamaGPTQForCausalLM
from transformers import LlamaForCausalLM

# define gpu settings as min_memory_per_gpu_in_gb -> settings

gpu_settings_1x24 = {0: {"max_memory": {0: f"{999*1024:.0f}MiB"}, "device_map": "auto"}}
gpu_settings_2x24 = {
    0: {"max_memory": {0: f"{17*1024:.0f}MiB", 1: f"{19*1024:.0f}MiB"}, "device_map": "auto"},
    26: {"max_memory": {0: f"{999*1024:.0f}MiB"}, "device_map": "auto"},
}


_llama_refs = LlamaGPTQForCausalLM, LlamaForCausalLM


llama_defaults = {
    "loader": "gptq",
    "loader_kwargs": {
        "inject_fused_attention": False,
        "disable_exllamav2": True,
        "use_safetensors": True,
        # disable_exllama=True,
        # use_triton=True,  # seems triton is not supported
    },
    "tokenizer_kwargs": {
        "use_fast": True,
    },
}

MODEL_DICT = {
    "llama1-30b-4bit": {
        "name": "TheBloke/LLaMa-30B-GPTQ",
        "gpu_settings": gpu_settings_1x24,
        **llama_defaults,
    },
    "llama2-70b-4bit": {
        "name": "TheBloke/Llama-2-70B-GPTQ",
        "gpu_settings": gpu_settings_2x24,
        **llama_defaults,
    },
    "llama2-70b-chat-4bit": {
        "name": "TheBloke/Llama-2-70B-Chat-GPTQ",
        "max_memory": {0: f"{17*1024:.0f}MiB", 1: f"{19*1024:.0f}MiB"},
        "gpu_settings": gpu_settings_2x24,
        **llama_defaults,
    },
}
