"""
Sample graphs from a trained GraphRNN checkpoint.

Usage:
    python -m src.generate --checkpoint outputs/checkpoints/community_rnn_epoch3000.pt --num_graphs 50
"""

import argparse
import os
import pickle
import torch
import networkx as nx
import matplotlib.pyplot as plt

from src.models.GraphRNN.model import GraphRNN
from src.models.GraphRNN.utils import sequence_to_graph

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_graphs", type=int, default=50)
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args=parse_args()
    device=torch.device(args.device)
    ckpt=torch.load(args.checkpoint, map_location=device)
    cfg=ckpt["args"]
    max_num_node=ckpt["max_num_node"]
    max_prev_node=ckpt["max_prev_node"]

    model=GraphRNN(
        max_prev_node=max_prev_node, graph_hidden=cfg["graph_hidden"],
        graph_embed=cfg["graph_embed"], edge_hidden=cfg["edge_hidden"], 
        edge_embed=cfg["edge_embed"], variant=cfg["variant"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sequences=model.generate(args.num_graphs, max_num_node, device=device)
    graphs=[sequence_to_graph(S) for S in sequences]
    graphs=[G for G in graphs if G.number_of_nodes() > 0]

    dataset_name=cfg["dataset"]
    variant=cfg["variant"]

    results_dir=os.path.join(args.out_dir, "results")
    figures_dir=os.path.join(args.out_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    out_pkl=os.path.join(results_dir, f"{dataset_name}_{variant}_generated.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(graphs, f)
    print(f"Generated {len(graphs)} graphs to {out_pkl}")

    n_show=min(8, len(graphs))
    if n_show > 0:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, G in zip(axes.flat, graphs[:n_show]):
            pos = nx.spring_layout(G, seed=0)
            nx.draw(G, pos, ax=ax, node_size=30, width=0.5, with_labels=False)
            ax.set_title(f"n={G.number_of_nodes()}, m={G.number_of_edges()}")
        for ax in axes.flat[n_show:]:
            ax.axis("off")
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f"{dataset_name}_{variant}_generated.png")
        plt.savefig(fig_path)
        print(f"Saved visualization -> {fig_path}")

if __name__=="__main__":
    main()