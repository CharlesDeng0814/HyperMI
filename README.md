# HyperMI Repository

This repository reorganizes the core code from `DualChannel_1 (2).ipynb` into a GitHub-friendly project structure while following the top-level layout of [CharlesDeng0814/HyperMI](https://github.com/CharlesDeng0814/HyperMI).

## Project Structure

```text
HyperMI/
|- Data/
|- output/
|- main.py
|- models.py
|- train_pred.py
|- utils.py
|- requirements.txt
`- README.md
```

## File Overview

- `main.py`: command-line entry for training, evaluation, and prediction export.
- `models.py`: hypergraph encoder, classifier, and dual-channel fusion model.
- `train_pred.py`: cross-validation, feature transformation, training, fusion evaluation, and OOF score aggregation logic.
- `utils.py`: data loading, hypergraph processing, metrics, and reproducibility helpers.
- `Data/`: expected input data directory.
- `output/`: prediction outputs.

## Data Files

The repository is currently organized around these files under `./Data/`:

- `796true.txt`
- `2187false.txt`
- `geneList.csv`
- `multiOmicsFeature.csv`
- `fullC2_genes.csv`
- `fullC5_genes.csv`
- `C2_hypergraph.csv`
- `C2_weights.csv`
- `C5_hypergraph.csv`
- `C5_weights.csv`

The original desktop experiment files such as `C2_hypergraph_mean_new.csv` and `C5_weights_mean_new.csv` were normalized into the naming scheme above for easier reuse and GitHub publishing.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training and export ranked predictions:

```bash
python main.py ./output/predictions.tsv
```

If you want to preserve the original C2 edge weights instead of replacing them with ones:

```bash
python main.py ./output/predictions.tsv --c2-weight-mode original
```

## Output

The current code reports only the final fusion performance:

- `Fusion AUROC`
- `Fusion AUPRC`

It does not report separate C2 or C5 performance metrics.

The ranked prediction scores are written to the path you pass on the command line, for example `./output/predictions.tsv`.

The exported per-gene score follows the OOF rule used in your final experiment:

- For genes that appear in the labeled cross-validation splits, the exported score is the `oof_mean` score aggregated from the folds where that gene was in the test split.
- For genes that never receive an OOF score, the exported score falls back to the full-model ensemble mean prediction across all seed-fold runs.

So the final output remains one score per gene, but labeled training genes use OOF predictions rather than in-fold fitted scores.

## Notes

- The repository layout is aligned with the public `HyperMI` repository, but the implementation is rebuilt from your notebook.
- The current code keeps the notebook's final dual-channel training strategy, including the random-forest one-hot feature transform, late fusion, and OOF-based final scoring.
- GPU is used automatically when available; otherwise the code falls back to CPU.
- On this machine, a local Python environment issue currently breaks `numpy` import at runtime, so full end-to-end execution may require fixing the Python environment first.
