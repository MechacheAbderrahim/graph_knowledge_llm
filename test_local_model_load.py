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


    if True:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=True,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=True,
            ).to(device)

    print("Device map:", getattr(model, "hf_device_map", None))

    prompt = "Return only this JSON: {\"ok\": true}"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    first_device = next(model.parameters()).device
    inputs = tokenizer([text], return_tensors="pt").to(first_device)
    output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    generated = output[0][inputs.input_ids.shape[1] :]

    print("Generated :", tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
