from kg_sections.section_1_data import (
    category_name_lookup,
    check_gpu,
    load_config,
    load_category_names,
    load_products,
    make_output_dirs,
    print_dataset_summary,
    resolve_categories,
)
from kg_sections.section_3_ontology import load_qwen_model
from kg_sections.section_4_extraction import run_categories
from kg_sections.section_5_postprocess import merge_all_category_graphs


CONFIG_PATH = "config.yaml"


def main():
    config = load_config(CONFIG_PATH)
    paths = config["paths"]
    model_config = config["model"]
    run_config = config["run"]
    ontology_config = config["ontology"]
    ontology_limits = config["ontology_limits"]
    generation_config = config.get("generation", {})
    execution_config = config.get("execution", {})

    print("Mode config :", config["_mode"])

    onto_dir, kg_dir = make_output_dirs(paths["ontology_dir"], paths["kg_dir"])

    check_gpu()

    df = load_products(paths["products_csv"])
    id_to_name = load_category_names(paths["categories_csv"])
    name_for_category = category_name_lookup(id_to_name)
    categories = resolve_categories(df, run_config["categories"])
    print_dataset_summary(df, categories)

    tokenizer, model = load_qwen_model(
        model_config["name"],
        load_in_4bit=model_config.get("load_in_4bit", False),
        offline=execution_config.get("offline", False),
    )

    summary = run_categories(
        categories,
        df,
        name_for_category,
        run_config["sample_size"],
        run_config["max_products_per_category"],
        run_config["seed"],
        onto_dir,
        kg_dir,
        tokenizer,
        model,
        allow_fallback_ontology=run_config.get("allow_fallback_ontology", False),
        force_regenerate_ontology=ontology_config.get("force_regenerate", False),
        ontology_max_new_tokens=ontology_config.get("max_new_tokens", 2200),
        ontology_limits=ontology_limits,
        deterministic_generation=generation_config.get("deterministic", True),
    )
    print(summary)

    merge_all_category_graphs(kg_dir, output_path=paths["global_kg_path"])


if __name__ == "__main__":
    main()
