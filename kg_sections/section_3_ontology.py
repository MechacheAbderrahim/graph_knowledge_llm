import json
import re
from pathlib import Path

from kg_sections.section_0_llm import ask_qwen, load_qwen_model
from kg_sections.section_2_sampling import representative_titles_by_category


DEFAULT_ONTOLOGY_LIMITS = {
    "max_classes": 3,
    "max_object_properties": 8,
    "max_data_properties": 5,
    "max_controlled_vocab": 5,
    "max_vocab_values": 8,
    "max_rules": 6,
}

def normalize_ontology_limits(ontology_limits=None):
    limits = DEFAULT_ONTOLOGY_LIMITS.copy()
    if ontology_limits:
        for key, value in ontology_limits.items():
            if value is not None:
                limits[key] = int(value)
    return limits


def build_meta_prompt(ontology_limits=None, global_ontology=None):
    limits = normalize_ontology_limits(ontology_limits)
    global_block = ""
    if global_ontology:
        global_block = f"""

GLOBAL_ONTOLOGY to follow:
{json.dumps(global_ontology, ensure_ascii=False, separators=(",", ":"))}

You MUST make the category ontology compatible with GLOBAL_ONTOLOGY:
- reuse global class and predicate names when they fit;
- add category-specific classes/properties only when the global ontology is too generic;
- do not create synonyms for existing global concepts."""

    return f"""You are an ontology designer for e-commerce product knowledge graphs.
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
{global_block}

Then ADD category-specific elements:
- 1-{limits['max_classes']} Product subclasses ONLY if the category mixes device vs accessory; else one class.
- object properties as PREDICATE (Domain -> Range).
- data properties (literals on the product).
- controlled vocabularies (closed lists) for the categorical slots.
- 4 to {limits['max_rules']} short extraction rules.

Output ONLY a JSON object with keys:
{{"category","classes","object_properties","data_properties","controlled_vocab","rules"}}
Keep the JSON compact:
- maximum {limits['max_classes']} classes
- maximum {limits['max_object_properties']} object_properties
- maximum {limits['max_data_properties']} data_properties
- maximum {limits['max_controlled_vocab']} controlled_vocab entries, each with maximum {limits['max_vocab_values']} values
- maximum {limits['max_rules']} rules
No prose, no code fences."""


def build_global_ontology_prompt(ontology_limits=None):
    limits = normalize_ontology_limits(ontology_limits)

    return f"""You are an ontology designer for a large e-commerce product catalog.
You receive DIVERSE product title samples from MULTIPLE categories.
Design ONE compact GLOBAL ontology used as the shared schema for all categories.

Focus ONLY on cross-category concepts that should stay consistent when all category
knowledge graphs are merged.

Mandatory backbone:
- Each product node id = "asin:<ASIN>", class = Product or a subclass of Product.
- All entity node ids use "type:CanonicalSlug" so identical entities merge.
- Include class Product.
- Include class Brand and predicate HAS_BRAND (Product -> Brand).
- Reuse common predicate verbs where they fit: HAS_TYPE, COMPATIBLE_WITH, FOR_AUDIENCE,
  HAS_FEATURE, MADE_OF, BUNDLES, INSTALLED_AT.
- OPEN-WORLD: if an attribute is absent, omit it. NEVER create catch-all nodes like
  Unbranded/Unknown/Other.

Output ONLY a JSON object with keys:
{{"category","classes","object_properties","data_properties","controlled_vocab","rules"}}
Use "GLOBAL" as category.
Keep the JSON compact:
- maximum {limits['max_classes']} classes
- maximum {limits['max_object_properties']} object_properties
- maximum {limits['max_data_properties']} data_properties
- maximum {limits['max_controlled_vocab']} controlled_vocab entries, each with maximum {limits['max_vocab_values']} values
- maximum {limits['max_rules']} rules
No prose, no code fences."""

REQUIRED_KEYS = {
    "category",
    "classes",
    "object_properties",
    "data_properties",
    "controlled_vocab",
    "rules",
}


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


def design_ontology(
    category_name,
    sample_titles,
    tokenizer,
    model,
    max_new_tokens=2200,
    ontology_limits=None,
    deterministic=True,
    global_ontology=None,
):
    user_prompt = "Category: " + category_name + "\nSample titles:\n"
    user_prompt += "\n".join("- " + title for title in sample_titles)
    raw = ask_qwen(
        build_meta_prompt(ontology_limits, global_ontology=global_ontology) + "\n\n" + user_prompt,
        tokenizer,
        model,
        max_new_tokens=max_new_tokens,
        deterministic=deterministic,
    )
    return extract_json(raw)


def design_global_ontology(
    category_samples,
    tokenizer,
    model,
    max_new_tokens=2200,
    ontology_limits=None,
    deterministic=True,
):
    user_prompt = "Global catalog sample:\n"
    for sample in category_samples:
        user_prompt += (
            f"\nCategory {sample['category_id']} ({sample['category_name']}):\n"
        )
        user_prompt += "\n".join("- " + title for title in sample["titles"])
        user_prompt += "\n"

    raw = ask_qwen(
        build_global_ontology_prompt(ontology_limits) + "\n\n" + user_prompt,
        tokenizer,
        model,
        max_new_tokens=max_new_tokens,
        deterministic=deterministic,
    )
    ontology = extract_json(raw)
    ontology["category"] = "GLOBAL"
    return ontology


def validate_ontology(ontology):
    missing = REQUIRED_KEYS - set(ontology)
    if missing:
        raise ValueError(f"ontologie incomplete, cles manquantes: {missing}")

    if "HAS_BRAND" not in json.dumps(ontology["object_properties"]):
        print("  HAS_BRAND manquant dans l'ontologie generee")

    return True


def build_extraction_prompt(ontology, global_ontology=None):
    global_block = ""
    if global_ontology:
        global_block = f"""
GLOBAL ONTOLOGY:
{json.dumps(global_ontology, ensure_ascii=False, indent=2)}
"""

    return f"""You are a KG extraction engine for the "{ontology['category']}" category.

{global_block}
CATEGORY ONTOLOGY:
{json.dumps(ontology, ensure_ascii=False, indent=2)}

FRAGMENT FORMAT:
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
- "edges" link the product to entities using ONLY the global/category ontology predicates.
- Follow the global ontology first, then use the category ontology for specific concepts.
- Omit anything absent from the title. NEVER output Unbranded/Unknown/Other.
- Output ONLY valid JSON, no prose, no code fences."""


def fallback_ontology(category_name):
    return {
        "category": category_name,
        "classes": ["Product"],
        "object_properties": [
            {"predicate": "HAS_BRAND", "domain": "Product", "range": "Brand"},
            {"predicate": "HAS_TYPE", "domain": "Product", "range": "ProductType"},
            {"predicate": "HAS_FEATURE", "domain": "Product", "range": "Feature"},
            {"predicate": "COMPATIBLE_WITH", "domain": "Product", "range": "CompatibleItem"},
            {"predicate": "MADE_OF", "domain": "Product", "range": "Material"},
            {"predicate": "FOR_AUDIENCE", "domain": "Product", "range": "Audience"},
        ],
        "data_properties": ["title", "price", "stars"],
        "controlled_vocab": {},
        "rules": [
            "Always create one product from asin, title, price, and stars.",
            "Extract a brand only when a brand is explicit in the title.",
            "Extract product type, features, compatibility, material, and audience when explicit.",
            "Omit absent or uncertain attributes.",
            "Never create Unknown, Other, or Unbranded nodes.",
        ],
    }


def get_global_ontology(
    categories,
    df,
    name_for_category,
    sample_size,
    onto_dir,
    tokenizer,
    model,
    force=False,
    allow_fallback=False,
    max_new_tokens=2200,
    ontology_limits=None,
    deterministic=True,
    sampling_text_column="title",
    output_path=None,
    seed=42,
):
    path = Path(output_path) if output_path else Path(onto_dir) / "global_ontology.json"
    if path.exists() and not force:
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    category_samples = representative_titles_by_category(
        categories,
        df,
        name_for_category,
        sample_size,
        seed=seed,
        text_column=sampling_text_column,
    )
    if not category_samples:
        raise ValueError("impossible de generer l'ontologie globale: aucun echantillon")

    last_error = None
    for attempt in range(2):
        try:
            ontology = design_global_ontology(
                category_samples,
                tokenizer,
                model,
                max_new_tokens=max_new_tokens,
                ontology_limits=ontology_limits,
                deterministic=deterministic,
            )
            validate_ontology(ontology)
            path.parent.mkdir(exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                json.dump(ontology, file, ensure_ascii=False, indent=2)
            return ontology
        except Exception as exc:
            last_error = exc
            print(f"  ontologie globale tentative {attempt + 1} echouee: {exc}")

    if allow_fallback:
        print("  utilisation d'une ontologie globale generique de test")
        ontology = fallback_ontology("GLOBAL")
        validate_ontology(ontology)
        path.parent.mkdir(exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(ontology, file, ensure_ascii=False, indent=2)
        return ontology

    raise RuntimeError(f"impossible de generer l'ontologie globale: {last_error}")


def get_ontology(
    category_id,
    category_name,
    titles,
    onto_dir,
    tokenizer,
    model,
    force=False,
    allow_fallback=False,
    max_new_tokens=2200,
    ontology_limits=None,
    deterministic=True,
    global_ontology=None,
):
    path = Path(onto_dir) / f"onto_{category_id}.json"
    if path.exists() and not force:
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    last_error = None
    for attempt in range(2):
        try:
            ontology = design_ontology(
                category_name,
                titles,
                tokenizer,
                model,
                max_new_tokens=max_new_tokens,
                ontology_limits=ontology_limits,
                deterministic=deterministic,
                global_ontology=global_ontology,
            )
            validate_ontology(ontology)
            with open(path, "w", encoding="utf-8") as file:
                json.dump(ontology, file, ensure_ascii=False, indent=2)
            return ontology
        except Exception as exc:
            last_error = exc
            print(f"  ontologie {category_id} tentative {attempt + 1} echouee: {exc}")

    if allow_fallback:
        print("  utilisation d'une ontologie generique de test")
        ontology = fallback_ontology(category_name)
        validate_ontology(ontology)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(ontology, file, ensure_ascii=False, indent=2)
        return ontology

    raise RuntimeError(f"impossible de generer l'ontologie pour {category_id}: {last_error}")
