import os
from copy import deepcopy


DEFAULT_MODEL_ROOT = "/lustre/fsmisc/dataset/HuggingFace_Models/"


def resolve_model_name(model_name, offline=False, model_root=DEFAULT_MODEL_ROOT):
    if offline:
        return os.path.join(model_root, model_name)
    return model_name


def get_best_device(torch):
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def dtype_for_device(torch, device):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def model_input_device(model):
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device
    return next(model.parameters()).device


def deterministic_generation_config(tokenizer, model):
    generation_config = deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None

    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = tokenizer.eos_token_id
    if generation_config.eos_token_id is None:
        generation_config.eos_token_id = tokenizer.eos_token_id

    return generation_config


def load_qwen_model(model_name, load_in_4bit=False, offline=False, model_root=DEFAULT_MODEL_ROOT):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        import accelerate  # noqa: F401

        has_accelerate = True
    except ImportError:
        has_accelerate = False
        print("accelerate absent -> installe accelerate puis redemarre le kernel.")

    resolved_model_name = resolve_model_name(
        model_name,
        offline=offline,
        model_root=model_root,
    )
    local_files_only = offline

    device = get_best_device(torch)
    dtype = dtype_for_device(torch, device)

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if load_in_4bit and device != "cuda":
        print("LOAD_IN_4BIT ignore : bitsandbytes 4-bit est reserve a CUDA.")
        load_in_4bit = False

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    if device == "cuda" and has_accelerate:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            device_map="auto",
            local_files_only=local_files_only,
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            local_files_only=local_files_only,
            **model_kwargs,
        ).to(device)

    print("Modele pret :", resolved_model_name, "| device:", model_input_device(model))
    print("Device map:", getattr(model, "hf_device_map", None))
    return tokenizer, model


def normalize_model_config(model_config):
    if isinstance(model_config, str):
        return {"name": model_config, "load_in_4bit": False}

    if not isinstance(model_config, dict):
        raise ValueError(f"configuration modele invalide: {model_config}")

    if "name" not in model_config:
        raise ValueError("chaque configuration modele doit contenir name")

    return {
        "name": model_config["name"],
        "load_in_4bit": bool(model_config.get("load_in_4bit", False)),
        "model_root": model_config.get("model_root", DEFAULT_MODEL_ROOT),
    }


def load_llm_roles(models_config, roles, offline=False):
    loaded_models = {}
    cache = {}

    for role in roles:
        if role not in models_config:
            raise ValueError(f"modele manquant dans config.yaml pour le role: {role}")

        model_config = normalize_model_config(models_config[role])
        cache_key = (
            model_config["name"],
            model_config["load_in_4bit"],
            offline,
            model_config["model_root"],
        )

        if cache_key not in cache:
            print(f"Chargement LLM [{role}] :", model_config["name"])
            cache[cache_key] = load_qwen_model(
                model_config["name"],
                load_in_4bit=model_config["load_in_4bit"],
                offline=offline,
                model_root=model_config["model_root"],
            )
        else:
            print(f"LLM reutilise [{role}] :", model_config["name"])

        loaded_models[role] = cache[cache_key]

    return loaded_models


def ask_qwen(prompt, tokenizer, model, max_new_tokens=1000, deterministic=True):
    messages = [{"role": "user", "content": prompt}]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer([text], return_tensors="pt").to(model_input_device(model))
    generation_kwargs = {"max_new_tokens": max_new_tokens}
    if deterministic:
        generation_kwargs["generation_config"] = deterministic_generation_config(tokenizer, model)

    output = model.generate(**inputs, **generation_kwargs)
    generated = output[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)
