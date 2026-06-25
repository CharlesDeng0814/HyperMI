from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def set_all_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def cal_auc(output: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    output_test = output.detach().cpu().numpy()
    output_test = np.exp(output_test)[:, 1]
    labels_test = labels.detach().cpu().numpy()
    auroc = roc_auc_score(labels_test, output_test)
    precision, recall, _ = precision_recall_curve(labels_test, output_test)
    auprc = auc(recall, precision)
    return auroc, auprc


def getData(
    positiveGenePath: str | Path,
    negativeGenePath: str | Path,
    geneList: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    positiveGene = pd.read_csv(positiveGenePath, header=None)[0].tolist()
    positiveGene = sorted(set(geneList) & set(positiveGene))

    negativeGene = pd.read_csv(negativeGenePath, header=None)[0].tolist()
    negativeGene = sorted(set(geneList) & set(negativeGene))
    
    labelFrame = pd.DataFrame(data=[0] * len(geneList), index=geneList)
    labelFrame.loc[positiveGene, :] = 1
    positiveIndex = np.where(labelFrame == 1)[0]
    labelFrame.loc[negativeGene, :] = -1
    negativeIndex = np.where(labelFrame == -1)[0]
    
    labelFrame = pd.DataFrame(data=[0] * len(geneList), index=geneList)
    labelFrame.loc[positiveGene, :] = 1
    
    sampleIndex = np.array(list(positiveIndex) + list(negativeIndex))
    label = np.array([1] * len(positiveIndex) + [0] * len(negativeIndex))
    return sampleIndex, label, labelFrame


def _resolve_first_existing(candidates: Iterable[str | Path]) -> Path:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    joined = "\n".join(f"- {Path(candidate)}" for candidate in candidates if candidate is not None)
    raise FileNotFoundError(f"None of the candidate files exist:\n{joined}")


def getHyperGraph(
    hypergraph_edges_path: str | Path,
    edges_weights_path: str | Path,
    genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hypergraph_edges = pd.read_csv(hypergraph_edges_path, index_col=None, header=None).values.T
    hypergraph_edges = torch.from_numpy(hypergraph_edges)

    edgesWeights = pd.read_csv(edges_weights_path, index_col=None, header=None).values.T[0]
    edgesWeights = torch.from_numpy(edgesWeights).unsqueeze(1).float()
    
    row = hypergraph_edges[0]
    col = hypergraph_edges[1]
    data = edgesWeights.squeeze(1)
    
    weighted = coo_matrix(
        (data, (row, col)),
        shape=(int(hypergraph_edges[0].max()) + 1, int(hypergraph_edges[1].max()) + 1),
    ).toarray()
    non_weighted = coo_matrix(
        (np.ones_like(edgesWeights.squeeze(1)), (row, col)),
        shape=(int(hypergraph_edges[0].max()) + 1, int(hypergraph_edges[1].max()) + 1),
    ).toarray()
    
    weighted_frame = pd.DataFrame(data=weighted, index=genes)
    non_weighted_frame = pd.DataFrame(data=non_weighted, index=genes)
    return weighted_frame, non_weighted_frame


def processingHypergraph(
    data_dir: str | Path = "./Data",
    c2_edges_path: str | Path | None = None,
    c2_weights_path: str | Path | None = None,
    c5_edges_path: str | Path | None = None,
    c5_weights_path: str | Path | None = None,
    device: torch.device | None = None,
):
    data_dir = Path(data_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    c2_genes = pd.read_csv(data_dir / "fullC2_genes.csv", header=None, index_col=None).iloc[:, 0].tolist()
    c5_genes = pd.read_csv(data_dir / "fullC5_genes.csv", header=None, index_col=None).iloc[:, 0].tolist()
    used_gene = pd.read_csv(data_dir / "geneList.csv", header=None, index_col=None).iloc[:, 0].tolist()
    
    c2_edges_path = _resolve_first_existing(
        [
            c2_edges_path,
            data_dir / "C2_hypergraph.csv",
            data_dir / "C2_hypergraph_edges.csv",
        ]
    )
    c2_weights_path = _resolve_first_existing([c2_weights_path, data_dir / "C2_weights.csv"])
    c5_edges_path = _resolve_first_existing(
        [
            c5_edges_path,
            data_dir / "C5_hypergraph.csv",
            data_dir / "C5_hypergraph_edges.csv",
            data_dir / "C5_hypergraph_mean_new.csv",
        ]
    )
    c5_weights_path = _resolve_first_existing(
        [c5_weights_path, data_dir / "C5_weights.csv", data_dir / "C5_weights_mean_new.csv"]
    )
    
    c2_weighted_frame, c2_non_weighted_frame = getHyperGraph(c2_edges_path, c2_weights_path, c2_genes)
    c5_weighted_frame, c5_non_weighted_frame = getHyperGraph(c5_edges_path, c5_weights_path, c5_genes)
    
    non_weighted_matrix = pd.concat([c2_non_weighted_frame, c5_non_weighted_frame], axis=1)
    non_weighted_matrix.columns = np.arange(non_weighted_matrix.shape[1])
    non_weighted_matrix = non_weighted_matrix.loc[used_gene].fillna(0)
    
    weighted_matrix = pd.concat([c2_weighted_frame, c5_weighted_frame], axis=1)
    weighted_matrix.columns = np.arange(weighted_matrix.shape[1])
    weighted_matrix = weighted_matrix.loc[used_gene].fillna(0)
    
    c2_non_weighted_frame = non_weighted_matrix.iloc[:, : c2_non_weighted_frame.shape[1]]
    c5_non_weighted_frame = non_weighted_matrix.iloc[:, c2_non_weighted_frame.shape[1] :]
    c2_weighted_frame = weighted_matrix.iloc[:, : c2_non_weighted_frame.shape[1]]
    c5_weighted_frame = weighted_matrix.iloc[:, c2_non_weighted_frame.shape[1] :]
    
    c2_hypergraph_edges = torch.nonzero(torch.from_numpy(c2_non_weighted_frame.values)).T
    c2_edges_weights = c2_weighted_frame.values[c2_hypergraph_edges[0], c2_hypergraph_edges[1]]
    c2_edges_weights = torch.from_numpy(c2_edges_weights).float()
    
    c5_hypergraph_edges = torch.nonzero(torch.from_numpy(c5_non_weighted_frame.values)).T
    c5_edges_weights = c5_weighted_frame.values[c5_hypergraph_edges[0], c5_hypergraph_edges[1]]
    c5_edges_weights = torch.from_numpy(c5_edges_weights).float()
    
    c2_data = (
        c2_non_weighted_frame,
        c2_hypergraph_edges.to(device),
        c2_edges_weights.to(device),
    )
    c5_data = (
        c5_non_weighted_frame,
        c5_hypergraph_edges.to(device),
        c5_edges_weights.to(device),
    )
    return c2_data, c5_data

