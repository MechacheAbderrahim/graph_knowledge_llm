import json
import re
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from kg_sections.section_0_llm import ask_qwen
from kg_sections.section_0_preprocessing import normalize_text, normalized_column_name
from kg_sections.section_2_sampling import representative_titles
from kg_sections.section_3_ontology import (
    build_extraction_prompt,
    extract_json,
    get_ontology,
)


PRODUCT_ID_COLUMN = "asin"
CATEGORY_ID_COLUMN = "category_id"
DIRECT_REL_OVERRIDES = {
    "brand": "HAS_BRAND",
    "brand_name": "HAS_BRAND",
    "manufacturer": "HAS_BRAND",
    CATEGORY_ID_COLUMN: "IN_CATEGORY",
}


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


def clean_value(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def canonical_slug(value):
    normalized = normalize_text(value)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "value"


def entity_type_for_column(column):
    parts = canonical_slug(column).split("_")
    return "".join(part.capitalize() for part in parts if part) or "Entity"


def predicate_for_direct_column(column):
    if column in DIRECT_REL_OVERRIDES:
        return DIRECT_REL_OVERRIDES[column]
    return "HAS_" + re.sub(r"[^A-Z0-9]+", "_", column.upper()).strip("_")


def is_literal_value(value):
    return isinstance(value, (int, float, bool))


def product_id_from_asin(asin):
    return f"asin:{asin}"


def product_from_row(row, columns_config=None, preprocessing_config=None):
    columns_config = columns_config or {}
    direct_columns = columns_config.get("direct", [])
    llm_columns = columns_config.get("llm", ["title"])
    asin = clean_value(row[PRODUCT_ID_COLUMN])

    product = {
        "asin": asin,
        "direct_columns": {},
        "llm_columns": {},
    }

    for column in direct_columns:
        if column in row:
            value = clean_value(row[column])
            if value is not None:
                product["direct_columns"][column] = value

    for column in llm_columns:
        if column in row:
            value = clean_value(row[column])
            if value is not None:
                product["llm_columns"][column] = value
                product[column] = value

        raw_column = f"{column}_raw"
        if raw_column in row:
            value = clean_value(row[raw_column])
            if value is not None:
                product["llm_columns"][raw_column] = value

        normalized_column = normalized_column_name(column, preprocessing_config)
        if normalized_column in row:
            value = clean_value(row[normalized_column])
            if value is not None:
                product["llm_columns"][normalized_column] = value
                product[normalized_column] = value

    return product


def product_data_properties(product_context=None):
    if not product_context:
        return {}

    properties = {}
    properties.update(product_context.get("direct_columns", {}))
    properties.update(product_context.get("llm_columns", {}))
    return properties


def upsert_product_node(nodes, product_id, data_properties=None, product_class=None):
    existing = nodes.get(product_id, {})
    existing_data_properties = existing.get("data_properties", {}).copy()
    existing_data_properties.update(data_properties or {})

    nodes[product_id] = {
        "id": product_id,
        "type": "Product",
        "class": product_class or existing.get("class") or "Product",
        "label": existing_data_properties.get("title", existing.get("label", "")),
        "data_properties": existing_data_properties,
    }


def merge_direct_columns(product_context, category_id, category_name, nodes, edges, columns_config=None):
    product_id = product_id_from_asin(product_context["asin"])
    direct_columns = product_context.get("direct_columns", {})

    upsert_product_node(
        nodes,
        product_id,
        data_properties=product_data_properties(product_context),
    )

    if CATEGORY_ID_COLUMN in direct_columns:
        category_node_id = f"category:{direct_columns[CATEGORY_ID_COLUMN]}"
        nodes.setdefault(
            category_node_id,
            {
                "id": category_node_id,
                "type": "Category",
                "label": category_name,
                "data_properties": {"category_id": int(category_id)},
            },
        )
        edge = {"from": product_id, "rel": "IN_CATEGORY", "to": category_node_id}
        edges[(edge["from"], edge["rel"], edge["to"])] = edge

    for column, value in direct_columns.items():
        if column in {PRODUCT_ID_COLUMN, CATEGORY_ID_COLUMN} or is_literal_value(value):
            continue

        node_id = f"{canonical_slug(column)}:{canonical_slug(value)}"
        node = {
            "id": node_id,
            "type": entity_type_for_column(column),
            "label": str(value),
        }
        nodes.setdefault(node_id, node)

        edge = {
            "from": product_id,
            "rel": predicate_for_direct_column(column),
            "to": node_id,
        }
        edges[(edge["from"], edge["rel"], edge["to"])] = edge


def merge_fragment(fragment, asin, nodes, edges, product_context=None):
    if "product" not in fragment:
        raise ValueError(f"cle 'product' absente (cles: {list(fragment)})")

    product = fragment["product"]
    product_id = product_id_from_asin(asin)
    data_properties = product.get("data_properties", {}).copy()
    data_properties.update(product_data_properties(product_context))

    upsert_product_node(
        nodes,
        product_id,
        data_properties=data_properties,
        product_class=product.get("class"),
    )

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


def merge_response(parsed, products, nodes, edges, category_id, category_name, columns_config=None):
    fragments = fragments_from_response(parsed)
    batch_asins = [product["asin"] for product in products]
    if len(fragments) != len(batch_asins):
        raise ValueError(
            f"nombre de fragments inattendu: {len(fragments)} pour {len(batch_asins)} produit(s)"
        )

    for index, fragment in enumerate(fragments):
        product_context = products[index]
        merge_direct_columns(
            product_context,
            category_id,
            category_name,
            nodes,
            edges,
            columns_config=columns_config,
        )
        merge_fragment(
            fragment,
            batch_asins[index],
            nodes,
            edges,
            product_context=product_context,
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
    preprocessing_config=None,
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
        products = [
            product_from_row(
                row,
                columns_config=columns_config,
                preprocessing_config=preprocessing_config,
            )
            for _, row in batch_df.iterrows()
        ]
        batch_asins = [product["asin"] for product in products]
        for product in products:
            merge_direct_columns(
                product,
                category_id,
                category_name,
                nodes,
                edges,
                columns_config=columns_config,
            )

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
            merge_response(
                extract_json(raw),
                products,
                nodes,
                edges,
                category_id,
                category_name,
                columns_config=columns_config,
            )
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
    preprocessing_config=None,
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
                preprocessing_config=preprocessing_config,
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
