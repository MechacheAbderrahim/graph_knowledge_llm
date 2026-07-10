import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from kg_sections.section_2_sampling import representative_titles
from kg_sections.section_3_ontology import (
    ask_qwen,
    build_extraction_prompt,
    extract_json,
    get_ontology,
)


def generate_product_prompt(category_prompt, product):
    return f"""{category_prompt}

Now extract the knowledge graph for THIS product.
Output ONLY the JSON object (product/nodes/edges), nothing else:
{json.dumps(product, ensure_ascii=False)}
"""


def product_from_row(row):
    return {
        "asin": row["asin"],
        "title": row["title"],
        "price": float(row["price"]),
        "stars": float(row["stars"]),
    }


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


def process_category(
    category_id,
    df,
    category_name,
    sample_size,
    max_products_per_cat,
    seed,
    onto_dir,
    kg_dir,
    tokenizer,
    model,
    allow_fallback_ontology=False,
    force_regenerate_ontology=False,
    ontology_max_new_tokens=2200,
    ontology_limits=None,
    deterministic_generation=True,
):
    category_df = df[df["category_id"] == category_id].reset_index(drop=True)
    if len(category_df) == 0:
        print("  (aucun produit)")
        return None

    titles = representative_titles(category_df, sample_size, seed=seed)
    ontology = get_ontology(
        category_id,
        category_name,
        titles,
        onto_dir,
        tokenizer,
        model,
        force=force_regenerate_ontology,
        allow_fallback=allow_fallback_ontology,
        max_new_tokens=ontology_max_new_tokens,
        ontology_limits=ontology_limits,
        deterministic=deterministic_generation,
    )
    category_prompt = build_extraction_prompt(ontology)

    work = category_df if max_products_per_cat is None else category_df.head(max_products_per_cat)
    nodes = {}
    edges = {}
    failures = []

    for _, row in tqdm(work.iterrows(), total=len(work), desc=f"cat {category_id}", leave=False):
        product = product_from_row(row)
        raw = ask_qwen(
            generate_product_prompt(category_prompt, product),
            tokenizer,
            model,
            deterministic=deterministic_generation,
        )

        try:
            merge_fragment(extract_json(raw), row["asin"], nodes, edges)
        except Exception as exc:
            failures.append({"asin": row["asin"], "error": str(exc), "raw": raw[:300]})

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
    tokenizer,
    model,
    allow_fallback_ontology=False,
    force_regenerate_ontology=False,
    ontology_max_new_tokens=2200,
    ontology_limits=None,
    deterministic_generation=True,
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
                tokenizer,
                model,
                allow_fallback_ontology=allow_fallback_ontology,
                force_regenerate_ontology=force_regenerate_ontology,
                ontology_max_new_tokens=ontology_max_new_tokens,
                ontology_limits=ontology_limits,
                deterministic_generation=deterministic_generation,
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
