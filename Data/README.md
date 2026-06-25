# Data Directory

This folder contains the data files expected by the current HyperMI codebase.

Included files:

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

The runtime reports only final fusion performance and exports one final score per gene.
For labeled genes, that final score is based on OOF predictions; for other genes, it falls back to the ensemble mean prediction.
