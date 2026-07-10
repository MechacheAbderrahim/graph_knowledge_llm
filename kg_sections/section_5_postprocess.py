import glob
import itertools
import json
from pathlib import Path

from kg_sections.section_2_sampling import representative_titles
from kg_sections.section_3_ontology import (
    ask_qwen,
    build_extraction_prompt,
    extract_json,
    get_ontology,
)
from kg_sections.section_4_extraction import generate_product_prompt, product_from_row


def debug_one_product(
    category_id,
    df,
    category_name,
    sample_size,
    seed,
    onto_dir,
    tokenizer,
    model,
    max_chars=1500,
):
    category_df = df[df["category_id"] == category_id].reset_index(drop=True)
    ontology = get_ontology(
        category_id,
        category_name,
        representative_titles(category_df, sample_size, seed=seed),
        onto_dir,
        tokenizer,
        model,
    )

    product = product_from_row(category_df.iloc[0])
    raw = ask_qwen(
        generate_product_prompt(build_extraction_prompt(ontology), product),
        tokenizer,
        model,
    )
    parsed = extract_json(raw)

    print("=== SORTIE BRUTE ===\n", raw[:max_chars])
    print("\n=== CLES PARSEES ===", list(parsed))
    return parsed


def merge_all_category_graphs(kg_dir, output_path="kg_global.json"):
    all_nodes = {}
    all_edges = {}

    for file_path in glob.glob(str(Path(kg_dir) / "kg_category_*.json")):
        with open(file_path, encoding="utf-8") as file:
            kg = json.load(file)

        for node in kg["nodes"]:
            all_nodes.setdefault(node["id"], node)

        for edge in kg["edges"]:
            all_edges[(edge["from"], edge["rel"], edge["to"])] = edge

    global_kg = {"nodes": list(all_nodes.values()), "edges": list(all_edges.values())}
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(global_kg, file, ensure_ascii=False, indent=2)

    print("KG global :", len(global_kg["nodes"]), "noeuds,", len(global_kg["edges"]), "aretes")
    return global_kg


def load_category_graph(kg_dir, category_id):
    path = Path(kg_dir) / f"kg_category_{category_id}.json"
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def visualize_category_graph(kg_dir, category_id, max_products=3, seed=7):
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    kg = load_category_graph(kg_dir, category_id)

    graph = nx.DiGraph()
    for node in kg["nodes"]:
        graph.add_node(node["id"], **{key: value for key, value in node.items() if key != "id"})

    for edge in kg["edges"]:
        graph.add_edge(edge["from"], edge["to"], rel=edge["rel"])

    product_ids = [
        node_id
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "Product"
    ][:max_products]

    keep = set(product_ids)
    for product_id in product_ids:
        keep |= set(graph.successors(product_id))

    subgraph = graph.subgraph(keep)
    node_types = sorted({data.get("type", "Entity") for _, data in subgraph.nodes(data=True)})

    color_of = {"Product": "#ff7f0e"}
    color_cycle = itertools.cycle(plt.cm.tab20.colors)
    for node_type in node_types:
        color_of.setdefault(node_type, next(color_cycle))

    node_colors = [
        color_of[subgraph.nodes[node_id].get("type", "Entity")] for node_id in subgraph.nodes
    ]
    node_sizes = [
        1700 if subgraph.nodes[node_id].get("type") == "Product" else 1100
        for node_id in subgraph.nodes
    ]
    labels = {
        node_id: (subgraph.nodes[node_id].get("label") or node_id.split(":", 1)[-1])[:22]
        for node_id in subgraph.nodes
    }
    edge_labels = {(src, dst): data["rel"] for src, dst, data in subgraph.edges(data=True)}

    plt.figure(figsize=(13, 9))
    positions = nx.spring_layout(subgraph, k=1.1, seed=seed)
    nx.draw_networkx_nodes(
        subgraph,
        positions,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=1.5,
    )
    nx.draw_networkx_edges(
        subgraph,
        positions,
        edge_color="#999",
        arrows=True,
        arrowsize=14,
        width=1.4,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_labels(subgraph, positions, labels, font_size=8.5, font_weight="bold")
    nx.draw_networkx_edge_labels(
        subgraph,
        positions,
        edge_labels,
        font_size=7,
        font_color="#444",
    )
    plt.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=node_type,
                markerfacecolor=color_of[node_type],
                markersize=11,
            )
            for node_type in node_types
        ],
        loc="upper left",
        fontsize=8,
    )
    plt.title(f"KG de {len(product_ids)} produit(s) - categorie {category_id}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return subgraph
