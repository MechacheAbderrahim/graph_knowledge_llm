import json
import re
from pathlib import Path


META_PROMPT = """You are an ontology designer for e-commerce product knowledge graphs.
You receive a category name and a DIVERSE sample of product titles from ONE category.
Design a compact ontology + extraction rules to turn its products into KG fragments.

You MUST reuse this shared backbone (do NOT rename it) so all categories stay mergeable:
- Each product node id = "asin:<ASIN>", class = a subclass of Product.
- All node ids use "type:CanonicalSlug" so identical entities merge across products.
- Always include class Brand and predicate HAS_BRAND (Product -> Brand).
- Reuse common predicate verbs where they fit: HAS_TYPE, COMPATIBLE_WITH, FOR_AUDIENCE,
  HAS_FEATURE, MADE_OF, BUNDLES, INSTALLED_AT.
- OPEN-WORLD: if an attribute is absent from a title, omit it. NEVER create catch-all
  nodes like Unbranded/Unknown/Other -> omit the edge instead.

Then ADD category-specific elements:
- 1-3 Product subclasses ONLY if the category mixes device vs accessory; else one class.
- object properties as PREDICATE (Domain -> Range).
- data properties (literals on the product).
- controlled vocabularies (closed lists) for the categorical slots.
- 4 to 8 short extraction rules.

Output ONLY a JSON object with keys:
{"category","classes","object_properties","data_properties","controlled_vocab","rules"}
No prose, no code fences."""

REQUIRED_KEYS = {
    "category",
    "classes",
    "object_properties",
    "data_properties",
    "controlled_vocab",
    "rules",
}


def load_qwen_model(model_name, load_in_4bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        import accelerate  # noqa: F401

        has_accelerate = True
    except ImportError:
        has_accelerate = False
        print("accelerate absent -> installe accelerate puis redemarre le kernel.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    if has_accelerate:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", **model_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)

    print("Modele pret :", model_name, "| device:", model.device)
    return tokenizer, model


def ask_qwen(prompt, tokenizer, model, max_new_tokens=1000):
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

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def extract_json(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]

    text = re.sub(r"^```[a-zA-Z]*", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("aucun JSON dans la sortie")

    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise ValueError("JSON non equilibre (tronque -> augmente max_new_tokens)")


def design_ontology(category_name, sample_titles, tokenizer, model):
    user_prompt = "Category: " + category_name + "\nSample titles:\n"
    user_prompt += "\n".join("- " + title for title in sample_titles)
    raw = ask_qwen(META_PROMPT + "\n\n" + user_prompt, tokenizer, model, max_new_tokens=1200)
    return extract_json(raw)


def validate_ontology(ontology):
    missing = REQUIRED_KEYS - set(ontology)
    if missing:
        raise ValueError(f"ontologie incomplete, cles manquantes: {missing}")

    if "HAS_BRAND" not in json.dumps(ontology["object_properties"]):
        print("  HAS_BRAND manquant dans l'ontologie generee")

    return True


def build_extraction_prompt(ontology):
    return f"""You are a KG extraction engine for the "{ontology['category']}" category.

ONTOLOGY:
{json.dumps(ontology, ensure_ascii=False, indent=2)}

OUTPUT FORMAT - return EXACTLY this shape, with these three keys:
{{
  "product": {{"id": "asin:<ASIN>", "class": "<a class from the ontology>",
               "data_properties": {{"title": "...", "price": 0.0, "stars": 0.0}}}},
  "nodes": [{{"id": "type:CanonicalSlug", "type": "<EntityType>", "label": "..."}}],
  "edges": [{{"from": "asin:<ASIN>", "rel": "<PREDICATE>", "to": "type:CanonicalSlug"}}]
}}

Example:
{{"product": {{"id": "asin:B000", "class": "ProjectorDevice",
   "data_properties": {{"title": "Mini 1080P Projector", "price": 79.99, "stars": 4.4}}}},
 "nodes": [{{"id": "brand:Acme", "type": "Brand", "label": "Acme"}},
           {{"id": "resolution:1080p_FHD", "type": "Resolution", "label": "1080p_FHD"}}],
 "edges": [{{"from": "asin:B000", "rel": "HAS_BRAND", "to": "brand:Acme"}},
           {{"from": "asin:B000", "rel": "HAS_RESOLUTION", "to": "resolution:1080p_FHD"}}]}}

RULES:
- The product goes in "product", NOT in "nodes".
- "nodes" = ONLY entities (brand, type, resolution...), each id = "type:Slug".
- "edges" link the product to entities using ONLY the ontology predicates.
- Omit anything absent from the title. NEVER output Unbranded/Unknown/Other.
- Output ONLY the JSON object, no prose, no code fences."""


def get_ontology(category_id, category_name, titles, onto_dir, tokenizer, model, force=False):
    path = Path(onto_dir) / f"onto_{category_id}.json"
    if path.exists() and not force:
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    last_error = None
    for attempt in range(2):
        try:
            ontology = design_ontology(category_name, titles, tokenizer, model)
            validate_ontology(ontology)
            with open(path, "w", encoding="utf-8") as file:
                json.dump(ontology, file, ensure_ascii=False, indent=2)
            return ontology
        except Exception as exc:
            last_error = exc
            print(f"  ontologie {category_id} tentative {attempt + 1} echouee: {exc}")

    raise RuntimeError(
        f"impossible de generer l'ontologie pour {category_id}: {last_error}"
    )
