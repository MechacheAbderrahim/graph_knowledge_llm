from pathlib import Path

import pandas as pd


def make_output_dirs(onto_dir="ontologies", kg_dir="kg"):
    onto_path = Path(onto_dir)
    kg_path = Path(kg_dir)
    onto_path.mkdir(exist_ok=True)
    kg_path.mkdir(exist_ok=True)
    return onto_path, kg_path


def check_gpu():
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("GPU         :", torch.cuda.get_device_name(0))
        print(f"VRAM totale : {props.total_memory / 1e9:.1f} GB")
        print("CUDA        :", torch.version.cuda)
    else:
        print("Pas de GPU - execution sur CPU (tres lent)")


def load_products(csv_path):
    useful_columns = ["asin", "title", "stars", "price", "category_id"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in useful_columns)
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
