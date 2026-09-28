"""
Train GraphRNN

Usage:
    python -m src.train --dataset community --variant rnn --epochs 3
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data import get_dataset, GraphSeqDataset
from src.model.model import GraphRNN
from src.model.utils import make_mask

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["community", "grid", "ego"], required=True)
    p.add_argument("--variant", choices=["rnn", "s"], default="rnn")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--graph_hidden", type=int, default=128)
    p.add_argument("--graph_embed", type=int, default=64)
    p.add_argument("--edge_hidden", type=int, default=16)
    p.add_argument("--edge_embed", type=int, default=8)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    args=parse_args()
    device=torch.device(args.device)
    torch.manual_seed(args.seed)

    train_graphs, test_graphs, max_num_node, max_prev_node=get_dataset(args.dataset, data_dir=args.data_dir, seed=args.seed)
    print(f"[{args.dataset}] train={len(train_graphs)} test={len(test_graphs)}" 
          f" max_num_node={max_num_node} max_prev_node={max_prev_node}")

    train_ds=GraphSeqDataset(train_graphs, max_prev_node, max_num_node)
    loader=DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model=GraphRNN(
        max_prev_node=max_prev_node, graph_hidden=args.graph_hidden,
        graph_embed=args.graph_embed, edge_hidden=args.edge_hidden, 
        edge_embed=args.edge_embed, variant=args.variant
    ).to(device)

    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.4), int(args.epochs*0.8)], gamma=0.3)
    bce=torch.nn.BCELoss(reduction="none")

    ckpt_dir=os.path.join(args.out_dir, "checkpoints")
    fig_dir=os.path.join(args.out_dir, "figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    losses=[]
    T=max_num_node-1
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss=0.0
        n_batches=0
        for x,y,lengths in loader:
            x,y=x.to(device), y.to(device)
            pred=model(x,y,lengths)
            mask=make_mask(lengths, T, max_prev_node, device=device)
            loss=(bce(pred,y)*mask).sum() / mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss +=loss.item()
            n_batches +=1
        scheduler.step()
        avg_loss=epoch_loss/max(1,n_batches)
        losses.append(avg_loss)

        if epoch%1==0 or epoch==1:
            print(f"epoch {epoch}/{args.epochs} loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.4f}")

        if epoch%args.save_every==0 or epoch==args.epochs:
            ckpt_path=os.path.join(ckpt_dir, f"{args.dataset}_{args.variant}_epoch_{epoch}.pt")
            torch.save({
                "model_state":model.state_dict(),
                "args":vars(args),
                "max_num_node":max_num_node,
                "max_prev_node":max_prev_node,
            }, ckpt_path)
            print(f" saved checkpoint -> {ckpt_path}")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("masked BCE loss")
    plt.title(f"GraphRNN-{args.variant} training on {args.dataset}")
    plt.savefig(os.path.join(fig_dir, f"{args.dataset}_{args.variant}_loss.pdf"))
    print("done.")

if __name__=="__main__":
    main()