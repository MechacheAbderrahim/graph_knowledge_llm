from pathlib import Path
from copy import deepcopy

import pandas as pd
import yaml


DEFAULT_COLUMNS = {
    "direct": ["asin", "category_id", "price", "stars"],
    "llm": ["title"],
}


def load_config(config_path="config.yaml"):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    validate_config(config)
    active_config = select_active_config(config)
    validate_active_config(active_config)
    return active_config


def validate_config(config):
    required_sections = ["paths", "models", "run", "ontology", "ontology_limits", "test"]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"sections manquantes dans config.yaml: {missing}")

    test = config.get("test", {})
    if "enabled" not in test:
        raise ValueError("config.yaml doit contenir test.enabled")

    validate_active_config(config)


def validate_active_config(config):
    columns = normalize_columns_config(config.get("columns"))
    if "title" not in columns["llm"]:
        raise ValueError("pour l'instant, columns.llm doit contenir title")

    required_models = ["global_ontology", "category_ontology", "product_kg"]
    missing_models = [role for role in required_models if role not in config.get("models", {})]
    if missing_models:
        raise ValueError(f"models incomplet dans config.yaml: {missing_models}")


def select_active_config(config):
    active = deepcopy(config)
    test = config.get("test", {})

    if test.get("enabled", False):
        for section in [
            "models",
            "run",
            "sampling",
            "ontology",
            "ontology_limits",
            "generation",
            "extraction",
            "resume",
            "columns",
            "preprocessing",
        ]:
            if section in test:
                active[section] = deepcopy(test[section])
        active["_mode"] = "test"
    else:
        active["_mode"] = "main"

    return active


def normalize_columns_config(columns_config=None):
    config = columns_config or {}
    direct_columns = list(config.get("direct", DEFAULT_COLUMNS["direct"]))
    llm_columns = list(config.get("llm", DEFAULT_COLUMNS["llm"]))

    return {
        "direct": direct_columns,
        "llm": llm_columns,
    }


def make_output_dirs(onto_dir="ontologies", kg_dir="kg"):
    onto_path = Path(onto_dir)
    kg_path = Path(kg_dir)
    onto_path.mkdir(exist_ok=True)
    kg_path.mkdir(exist_ok=True)
    return onto_path, kg_path


def get_best_device(torch):
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def check_gpu():
    import os

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    device = get_best_device(torch)

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print("GPU         :", torch.cuda.get_device_name(0))
        print(f"VRAM totale : {props.total_memory / 1e9:.1f} GB")
        print("CUDA        :", torch.version.cuda)
    elif device == "mps":
        print("GPU         : Apple Silicon / Metal")
        print("Backend     : MPS")
    else:
        print("Pas de GPU CUDA/MPS - execution sur CPU (tres lent)")


def load_products(csv_path, columns_config=None):
    columns = normalize_columns_config(columns_config)
    useful_columns = set(columns["direct"]) | set(columns["llm"]) | {"asin", "title", "category_id"}
    df = pd.read_csv(csv_path, usecols=lambda c: c in useful_columns)

    missing_required = [column for column in ["asin", "title", "category_id"] if column not in df]
    if missing_required:
        raise ValueError(f"colonnes obligatoires manquantes dans products_csv: {missing_required}")

    return df.dropna(subset=["title"]).reset_index(drop=True)


def load_category_names(categories_path):
    path = Path(categories_path)
    if not path.exists():
        return {}

    cats = pd.read_csv(path).rename(columns={"id": "category_id"})
    return dict(zip(cats["category_id"], cats["category_name"]))


def category_name_lookup(id_to_name):
    def name_for_category(category_id):
        return id_to_name.get(category_id, f"cat_{category_id}")

    return name_for_category


def resolve_categories(df, selected_categories):
    if selected_categories is None:
        return sorted(df["category_id"].unique().tolist())
    return selected_categories


def print_dataset_summary(df, categories):
    print(f"{len(df):,} produits | {df['category_id'].nunique()} categories au total")
    preview = categories[:10]
    suffix = " ..." if len(categories) > 10 else ""
    print(f"A traiter : {len(categories)} categorie(s) -> {preview}{suffix}")
