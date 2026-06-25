from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_pred import trainPred
from utils import processingHypergraph, set_all_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the HyperMI dual-channel training pipeline and export ranked predictions."
    )
    parser.add_argument("output_path", help="Path to the output ranking file.")
    parser.add_argument("--data-dir", default="./Data", help="Directory containing all input data files.")
    parser.add_argument("--positive-gene-path", default=None, help="Override path for positive genes.")
    parser.add_argument("--negative-gene-path", default=None, help="Override path for negative genes.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Fusion model learning rate.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Fusion model weight decay.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--n-hid", type=int, default=256, help="Hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.7, help="Fusion dropout.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for the 5 repeated CV runs.")
    parser.add_argument(
        "--c2-weight-mode",
        choices=["ones", "original"],
        default="ones",
        help="Whether to replace C2 edge weights with ones, following the final notebook experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    
    positive_gene_path = Path(args.positive_gene_path) if args.positive_gene_path else data_dir / "796true.txt"
    negative_gene_path = Path(args.negative_gene_path) if args.negative_gene_path else data_dir / "2187false.txt"
    
    geneList = pd.read_csv(data_dir / "geneList.csv", header=None, index_col=None).iloc[:, 0].tolist()
    multiFeature = pd.read_csv(data_dir / "multiOmicsFeature.csv", index_col=0)
    multiFeature = multiFeature.loc[geneList].values
    
    C2_data, C5_data = processingHypergraph(data_dir=data_dir)
    if args.c2_weight_mode == "ones":
        C2_data = (C2_data[0], C2_data[1], torch.ones_like(C2_data[2]))
    
    aurocList, auprcList, _predictionRes_full, _evaluationRes_oof, final_scores = trainPred(
        geneList,
        multiFeature,
        C2_data,
        C5_data,
        str(positive_gene_path),
        str(negative_gene_path),
        args.lr,
        args.epochs,
        args.dropout,
        args.n_hid,
        args.weight_decay,
        base_seed=args.seed,
    )
    
    predRes = final_scores["final_score"].sort_values(ascending=False)
    predRes.to_csv(output_path, sep="\t", header=False)
    
    print(f"Fusion AUROC: {np.mean(aurocList):.6f}")
    print(f"Fusion AUPRC: {np.mean(auprcList):.6f}")
    print(f"Saved ranked predictions to: {output_path}")


if __name__ == "__main__":
    main()
