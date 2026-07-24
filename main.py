from kg_sections.section_1_data import (
    category_name_lookup,
    check_gpu,
    load_config,
    load_category_names,
    load_products,
    make_output_dirs,
    normalize_columns_config,
    print_dataset_summary,
    resolve_categories,
)
from kg_sections.section_0_llm import load_llm_roles
from kg_sections.section_0_preprocessing import normalized_column_name, preprocess_products
from kg_sections.section_3_ontology import get_global_ontology
from kg_sections.section_4_extraction import run_categories
from kg_sections.section_5_postprocess import merge_all_category_graphs


CONFIG_PATH = "config.yaml"


def main():
    config = load_config(CONFIG_PATH)
    paths = config["paths"]
    models_config = config["models"]
    run_config = config["run"]
    sampling_config = config.get("sampling", {})
    ontology_config = config["ontology"]
    ontology_limits = config["ontology_limits"]
    generation_config = config.get("generation", {})
    execution_config = config.get("execution", {})
    extraction_config = config.get("extraction", {})
    columns_config = normalize_columns_config(config.get("columns"))
    preprocessing_config = config.get("preprocessing", {})

    print("Mode config :", config["_mode"])

    onto_dir, kg_dir = make_output_dirs(paths["ontology_dir"], paths["kg_dir"])

    check_gpu()

    df = load_products(paths["products_csv"], columns_config=columns_config)
    df = preprocess_products(
        df,
        preprocessing_config=preprocessing_config,
        llm_columns=columns_config["llm"],
    )
    id_to_name = load_category_names(paths["categories_csv"])
    name_for_category = category_name_lookup(id_to_name)
    categories = resolve_categories(df, run_config["categories"])
    print_dataset_summary(df, categories)

    llms = load_llm_roles(
        models_config,
        roles=["global_ontology", "category_ontology", "product_kg"],
        offline=execution_config.get("offline", False),
    )
    global_tokenizer, global_model = llms["global_ontology"]
    ontology_tokenizer, ontology_model = llms["category_ontology"]
    product_tokenizer, product_model = llms["product_kg"]

    sampling_text_column = "title"
    normalized_title = normalized_column_name("title", preprocessing_config)
    if normalized_title in df.columns:
        sampling_text_column = normalized_title

    global_ontology = get_global_ontology(
        categories,
        df,
        name_for_category,
        sampling_config.get("k_sample_global_onto", 3),
        onto_dir,
        global_tokenizer,
        global_model,
        force=ontology_config.get("force_regenerate", False),
        allow_fallback=run_config.get("allow_fallback_ontology", False),
        max_new_tokens=ontology_config.get("max_new_tokens", 2200),
        ontology_limits=ontology_limits,
        deterministic=generation_config.get("deterministic", True),
        sampling_text_column=sampling_text_column,
        output_path=paths.get("global_ontology_path"),
        seed=run_config["seed"],
    )
    print("Ontologie globale :", global_ontology.get("category", "GLOBAL"))

    summary = run_categories(
        categories,
        df,
        name_for_category,
        sampling_config.get("k_sample_category_onto", run_config.get("sample_size", 15)),
        run_config["max_products_per_category"],
        run_config["seed"],
        onto_dir,
        kg_dir,
        ontology_tokenizer,
        ontology_model,
        product_tokenizer,
        product_model,
        allow_fallback_ontology=run_config.get("allow_fallback_ontology", False),
        force_regenerate_ontology=ontology_config.get("force_regenerate", False),
        ontology_max_new_tokens=ontology_config.get("max_new_tokens", 2200),
        ontology_limits=ontology_limits,
        deterministic_generation=generation_config.get("deterministic", True),
        sampling_text_column=sampling_text_column,
        global_ontology=global_ontology,
        batch_size=extraction_config.get("batch_size", 1),
        product_max_new_tokens=extraction_config.get("max_new_tokens", 1000),
    )
    print(summary)

    merge_all_category_graphs(kg_dir, output_path=paths["global_kg_path"])


if __name__ == "__main__":
    main()
