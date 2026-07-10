import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = (
    "/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen2.5-3B-Instruct"
)


def best_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = best_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float16 if device == "mps" else torch.float32

    print("Model path:", model_path)
    print("Device    :", device)
    print("Offline   :", os.environ["TRANSFORMERS_OFFLINE"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)

    prompt = "Return only this JSON: {\"ok\": true}"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    generated = output[0][inputs.input_ids.shape[1] :]

    print("Generated :", tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
