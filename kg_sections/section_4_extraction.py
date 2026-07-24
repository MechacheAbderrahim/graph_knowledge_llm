import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from kg_sections.section_0_llm import ask_qwen
from kg_sections.section_2_sampling import representative_titles
from kg_sections.section_3_ontology import (
    build_extraction_prompt,
    extract_json,
    get_ontology,
)


def generate_product_prompt(category_prompt, product):
    return f"""{category_prompt}

Now extract the knowledge graph for THIS product.
Return ONLY one JSON fragment with keys: product, nodes, edges.

INPUT PRODUCT:
{json.dumps(product, ensure_ascii=False)}
"""


def generate_product_batch_prompt(category_prompt, products):
    return f"""{category_prompt}

Now extract the knowledge graph for this BATCH of products.
Return ONLY this JSON shape:
{{
  "fragments": [
    {{
      "product": {{"id": "asin:<ASIN>", "class": "<class>", "data_properties": {{...}}}},
      "nodes": [{{"id": "type:CanonicalSlug", "type": "<EntityType>", "label": "..."}}],
      "edges": [{{"from": "asin:<ASIN>", "rel": "<PREDICATE>", "to": "type:CanonicalSlug"}}]
    }}
  ]
}}

Rules:
- Return exactly one fragment per input product.
- Do not merge different products into one product object.
- Output ONLY the JSON object, no prose, no code fences.

INPUT PRODUCTS:
{json.dumps(products, ensure_ascii=False)}
"""


def product_from_row(row):
    product = {
        "asin": row["asin"],
        "title": row["title"],
    }

    if "title_normalized" in row:
        product["title_normalized"] = row["title_normalized"]
    if "title_raw" in row:
        product["title_raw"] = row["title_raw"]
    if "price" in row and pd.notna(row["price"]):
        product["price"] = float(row["price"])
    if "stars" in row and pd.notna(row["stars"]):
        product["stars"] = float(row["stars"])

    return product


def merge_fragment(fragment, asin, nodes, edges):
    if "product" not in fragment:
        raise ValueError(f"cle 'product' absente (cles: {list(fragment)})")

    product = fragment["product"]
    product_id = product.get("id") or f"asin:{asin}"
    data_properties = product.get("data_properties", {})

    nodes[product_id] = {
        "id": product_id,
        "type": "Product",
        "class": product.get("class"),
        "label": data_properties.get("title", ""),
        "data_properties": data_properties,
    }

    for node in fragment.get("nodes", []):
        if "id" in node:
            nodes.setdefault(node["id"], node)

    for edge in fragment.get("edges", []):
        if {"from", "rel", "to"} <= set(edge):
            edges[(edge["from"], edge["rel"], edge["to"])] = edge


def fragments_from_response(parsed):
    if "fragments" in parsed:
        fragments = parsed["fragments"]
        if not isinstance(fragments, list):
            raise ValueError("cle 'fragments' presente mais ce n'est pas une liste")
        return fragments

    if "product" in parsed:
        return [parsed]

    raise ValueError(f"format batch invalide (cles: {list(parsed)})")


def asin_from_fragment(fragment, fallback_asin):
    product_id = fragment.get("product", {}).get("id", "")
    if isinstance(product_id, str) and product_id.startswith("asin:"):
        return product_id.split(":", 1)[1]
    return fallback_asin


def merge_response(parsed, batch_asins, nodes, edges):
    fragments = fragments_from_response(parsed)
    if len(fragments) != len(batch_asins):
        raise ValueError(
            f"nombre de fragments inattendu: {len(fragments)} pour {len(batch_asins)} produit(s)"
        )

    for index, fragment in enumerate(fragments):
        merge_fragment(
            fragment,
            asin_from_fragment(fragment, batch_asins[index]),
            nodes,
            edges,
        )


def dataframe_batches(df, batch_size):
    batch_size = max(int(batch_size or 1), 1)
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def process_category(
    category_id,
    df,
    category_name,
    sample_size,
    max_products_per_cat,
    seed,
    onto_dir,
    kg_dir,
    ontology_tokenizer,
    ontology_model,
    product_tokenizer,
    product_model,
    allow_fallback_ontology=False,
    force_regenerate_ontology=False,
    ontology_max_new_tokens=2200,
    ontology_limits=None,
    deterministic_generation=True,
    sampling_text_column="title",
    global_ontology=None,
    batch_size=1,
    product_max_new_tokens=1000,
    columns_config=None,
):
    category_df = df[df["category_id"] == category_id].reset_index(drop=True)
    if len(category_df) == 0:
        print("  (aucun produit)")
        return None

    titles = representative_titles(
        category_df,
        sample_size,
        seed=seed,
        text_column=sampling_text_column,
    )
    ontology = get_ontology(
        category_id,
        category_name,
        titles,
        onto_dir,
        ontology_tokenizer,
        ontology_model,
        force=force_regenerate_ontology,
        allow_fallback=allow_fallback_ontology,
        max_new_tokens=ontology_max_new_tokens,
        ontology_limits=ontology_limits,
        deterministic=deterministic_generation,
        global_ontology=global_ontology,
    )
    category_prompt = build_extraction_prompt(
        ontology,
        global_ontology=global_ontology,
        columns_config=columns_config,
    )

    work = category_df if max_products_per_cat is None else category_df.head(max_products_per_cat)
    nodes = {}
    edges = {}
    failures = []

    total_batches = (len(work) + max(int(batch_size or 1), 1) - 1) // max(int(batch_size or 1), 1)
    for batch_df in tqdm(
        dataframe_batches(work, batch_size),
        total=total_batches,
        desc=f"cat {category_id}",
        leave=False,
    ):
        products = [product_from_row(row) for _, row in batch_df.iterrows()]
        batch_asins = [product["asin"] for product in products]
        prompt = (
            generate_product_prompt(category_prompt, products[0])
            if len(products) == 1
            else generate_product_batch_prompt(category_prompt, products)
        )
        raw = ask_qwen(
            prompt,
            product_tokenizer,
            product_model,
            max_new_tokens=product_max_new_tokens,
            deterministic=deterministic_generation,
        )

        try:
            merge_response(extract_json(raw), batch_asins, nodes, edges)
        except Exception as exc:
            failures.append({"asin": ",".join(batch_asins), "error": str(exc), "raw": raw[:300]})

    kg = {
        "category_id": int(category_id),
        "category_name": category_name,
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }

    path = Path(kg_dir) / f"kg_category_{category_id}.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(kg, file, ensure_ascii=False, indent=2)

    return {
        "category_id": category_id,
        "name": category_name,
        "products": len(work),
        "nodes": len(kg["nodes"]),
        "edges": len(kg["edges"]),
        "failures": len(failures),
        "failures_detail": failures,
    }


def run_categories(
    categories,
    df,
    name_for_category,
    sample_size,
    max_products_per_cat,
    seed,
    onto_dir,
    kg_dir,
    ontology_tokenizer,
    ontology_model,
    product_tokenizer,
    product_model,
    allow_fallback_ontology=False,
    force_regenerate_ontology=False,
    ontology_max_new_tokens=2200,
    ontology_limits=None,
    deterministic_generation=True,
    sampling_text_column="title",
    global_ontology=None,
    batch_size=1,
    product_max_new_tokens=1000,
    columns_config=None,
):
    summary = []

    for category_id in categories:
        category_name = name_for_category(category_id)
        print(f"\n=== Categorie {category_id} ({category_name}) ===")

        try:
            result = process_category(
                category_id,
                df,
                category_name,
                sample_size,
                max_products_per_cat,
                seed,
                onto_dir,
                kg_dir,
                ontology_tokenizer,
                ontology_model,
                product_tokenizer,
                product_model,
                allow_fallback_ontology=allow_fallback_ontology,
                force_regenerate_ontology=force_regenerate_ontology,
                ontology_max_new_tokens=ontology_max_new_tokens,
                ontology_limits=ontology_limits,
                deterministic_generation=deterministic_generation,
                sampling_text_column=sampling_text_column,
                global_ontology=global_ontology,
                batch_size=batch_size,
                product_max_new_tokens=product_max_new_tokens,
                columns_config=columns_config,
            )
            if result:
                summary.append(
                    {key: value for key, value in result.items() if key != "failures_detail"}
                )
                print(
                    f"  -> {result['nodes']} noeuds, "
                    f"{result['edges']} aretes, "
                    f"{result['failures']} echec(s)"
                )
                if result["failures"]:
                    print("     1er echec:", result["failures_detail"][0]["error"])
        except Exception as exc:
            print("  ERREUR categorie:", exc)

    return pd.DataFrame(summary)
