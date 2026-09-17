"""
Evaluate generated graphs against a reference (test) set using MMD
over degree and clustering-coefficient distributions (Sec 4.3 of the paper).
Orbit-count MMD is omitted (needs an external orbit counter, e.g. ORCA);
degree + clustering already give a solid quantitative signal.

Usage:
    python -m src.evaluate --dataset community --variant rnn
"""

import argparse
import json
import os
import pickle
import numpy as np
import networkx as nx
from scipy.stats import wasserstein_distance

def degree_seq(G):
    return [d for _, d in G.degree()]

def clustering_seq(G):
    return list(nx.clustering(G).values())

def gaussian_kernel(x, y, sigma=1.0):
    d=wasserstein_distance(x, y) if len(x) and len(y) else 0.0
    return np.exp(-d/(2*sigma**2))

def mmd(graphs1, graphs2, stat_fn, sigma=1.0):
    stats1=[stat_fn(G) or [0] for G in graphs1]
    stats2=[stat_fn(G) or [0] for G in graphs2]

    def mean_kernel(a, b):
        vals=[gaussian_kernel(x, y, sigma) for x in a for y in b]
        return float(np.mean(vals)) if vals else 0.0

    k11=mean_kernel(stats1, stats1)
    k22=mean_kernel(stats2, stats2)
    k12=mean_kernel(stats1, stats2)
    return k11+k22 - 2*k12

def evaluate(test_graphs, gen_graphs):
    return {
        "mmd_degree": mmd(test_graphs, gen_graphs, degree_seq),
        "mmd_clustering": mmd(test_graphs, gen_graphs, clustering_seq),
        "n_test": len(test_graphs),
        "n_generated": len(gen_graphs)
    }

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--variant", choices=["rnn", "s"], default="rnn")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()

def main():
    args=parse_args()

    test_path=os.path.join(args.data_dir, "processed", f"{args.dataset}.pkl")
    gen_path=os.path.join(args.out_dir, "results", f"{args.dataset}_{args.variant}_generated.pkl")

    with open(test_path, "rb") as f:
        all_graphs=pickle.load(f)
    test_graphs=all_graphs[int(0.8 * len(all_graphs)):]  # matches the train/test split in data.py

    with open(gen_path, "rb") as f:
        gen_graphs=pickle.load(f)

    metrics=evaluate(test_graphs, gen_graphs)
    print(json.dumps(metrics, indent=2))

    out_json = os.path.join(args.out_dir, "results", f"{args.dataset}_{args.variant}_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()