from kg_sections.section_1_data import (
    category_name_lookup,
    check_gpu,
    load_category_names,
    load_products,
    make_output_dirs,
    print_dataset_summary,
    resolve_categories,
)
from kg_sections.section_3_ontology import load_qwen_model
from kg_sections.section_4_extraction import run_categories
from kg_sections.section_5_postprocess import merge_all_category_graphs


CSV_PATH = "./amazon_products.csv"
CATEGORIES_PATH = "./amazon_categories.csv"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" #"Qwen/Qwen2.5-3B-Instruct"
LOAD_IN_4BIT = False

SAMPLE_SIZE = 15
MAX_PRODUCTS_PER_CAT = 25
SEED = 42
ALLOW_FALLBACK_ONTOLOGY = True

#CATEGORIES_TO_RUN = [77, 228, 83, 32]
CATEGORIES_TO_RUN = [77]
# CATEGORIES_TO_RUN = None # To Run all 

ONTO_DIR = "ontologies"
KG_DIR = "kg"


def main():
    onto_dir, kg_dir = make_output_dirs(ONTO_DIR, KG_DIR)

    check_gpu()

    df = load_products(CSV_PATH)
    id_to_name = load_category_names(CATEGORIES_PATH)
    name_for_category = category_name_lookup(id_to_name)
    categories = resolve_categories(df, CATEGORIES_TO_RUN)
    print_dataset_summary(df, categories)

    tokenizer, model = load_qwen_model(MODEL_NAME, load_in_4bit=LOAD_IN_4BIT)

    summary = run_categories(
        categories,
        df,
        name_for_category,
        SAMPLE_SIZE,
        MAX_PRODUCTS_PER_CAT,
        SEED,
        onto_dir,
        kg_dir,
        tokenizer,
        model,
        allow_fallback_ontology=ALLOW_FALLBACK_ONTOLOGY,
    )
    print(summary)

    merge_all_category_graphs(kg_dir, output_path="kg_global.json")


if __name__ == "__main__":
    main()
