from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from models import Classifier_1, NWHCEncoder, dualChannelArchitecture
from utils import cal_auc, getData, set_all_seeds


def _build_hyperedge_weight(non_weighted_frame: pd.DataFrame, train_positive_gene: list[str]) -> torch.Tensor:
    positive_matrix_sum = non_weighted_frame.loc[train_positive_gene].sum()
    weight = positive_matrix_sum.values
    used_col_sum = non_weighted_frame.values.sum(0) + 1e-8
    return torch.from_numpy(weight / used_col_sum).float()


def train_test(
    trainIndex,
    testIndex,
    labelFrame: pd.DataFrame,
    multi_feature: torch.Tensor,
    C2_data,
    C5_data,
    geneList: list[str],
    lr: float,
    epochs: int,
    dropout: float,
    n_hid: int,
    weight_decay: float,
):
    device = multi_feature.device
    trainFrame = labelFrame.iloc[trainIndex]
    trainPositiveGene = list(trainFrame.where(trainFrame == 1).dropna().index)

    c2_hyperedge_weight = _build_hyperedge_weight(C2_data[0], trainPositiveGene).to(device)
    c5_hyperedge_weight = _build_hyperedge_weight(C5_data[0], trainPositiveGene).to(device)
    
    model_C2 = NWHCEncoder(
        in_dim=multi_feature.shape[1],
        edge_dim=n_hid,
        node_dim=n_hid,
        num_layers=3,
        dropout=0.5,
        n_class=2,
        norm_mode="wm_ew",
    ).to(device)
    model_C5 = NWHCEncoder(
        in_dim=multi_feature.shape[1],
        edge_dim=n_hid,
        node_dim=n_hid,
        num_layers=3,
        dropout=0.3,
        n_class=2,
        norm_mode="wm_ew",
    ).to(device)
    
    classifier_C2 = Classifier_1(in_dim=n_hid, out_dim=2).to(device)
    classifier_C5 = Classifier_1(in_dim=n_hid, out_dim=2).to(device)
    model_fusion = dualChannelArchitecture(featureDim=n_hid, dropout=dropout).to(device)
    
    optimizer_C2 = optim.Adam(
        list(model_C2.parameters()) + list(classifier_C2.parameters()),
        lr=0.02,
        weight_decay=5e-4,
    )
    optimizer_C5 = optim.Adam(
        list(model_C5.parameters()) + list(classifier_C5.parameters()),
        lr=0.05,
        weight_decay=5e-4,
    )
    scheduler_C2 = optim.lr_scheduler.MultiStepLR(
        optimizer_C2,
        milestones=[100, 200, 300, 400],
        gamma=0.5,
    )
    scheduler_C5 = optim.lr_scheduler.MultiStepLR(
        optimizer_C5,
        milestones=[100, 200, 300, 400],
        gamma=0.5,
    )
    optimizer_fusion = optim.AdamW(model_fusion.parameters(), lr=lr, weight_decay=weight_decay)
    
    labels = torch.from_numpy(labelFrame.values.reshape(-1,)).long().to(device)
    
    for epoch in range(epochs):
        model_C2.train()
        model_C5.train()
        classifier_C2.train()
        classifier_C5.train()
        optimizer_C2.zero_grad()
        optimizer_C5.zero_grad()
    
        emb_C2 = model_C2(multi_feature, C2_data[1], c2_hyperedge_weight, C2_data[2])
        emb_C5 = model_C5(multi_feature, C5_data[1], c5_hyperedge_weight, C5_data[2])
        output_C2 = classifier_C2(emb_C2)
        output_C5 = classifier_C5(emb_C5)
    
        loss_train_C2 = F.nll_loss(output_C2[trainIndex], labels[trainIndex])
        loss_train_C5 = F.nll_loss(output_C5[trainIndex], labels[trainIndex])
        loss_train_C2.backward()
        loss_train_C5.backward()
        optimizer_C2.step()
        optimizer_C5.step()
        scheduler_C2.step()
        scheduler_C5.step()
    
        if epoch > epochs / 2:
            model_fusion.train()
            optimizer_fusion.zero_grad()
            feat_C2 = emb_C2.detach()
            feat_C5 = emb_C5.detach()
            output_fusion = model_fusion(feat_C2, feat_C5)
            loss_train_fusion = F.nll_loss(output_fusion[trainIndex], labels[trainIndex])
            loss_train_fusion.backward()
            optimizer_fusion.step()
    
    model_C2.eval()
    model_C5.eval()
    model_fusion.eval()
    classifier_C2.eval()
    classifier_C5.eval()
    
    with torch.no_grad():
        emb_C2_eval = model_C2(multi_feature, C2_data[1], c2_hyperedge_weight, C2_data[2])
        emb_C5_eval = model_C5(multi_feature, C5_data[1], c5_hyperedge_weight, C5_data[2])
        output_fusion = model_fusion(emb_C2_eval, emb_C5_eval)
        auroc_val, auprc_val = cal_auc(output_fusion[testIndex], labels[testIndex])
    
        full_prob = output_fusion.exp().detach().cpu().numpy()[:, 1]
        test_prob = full_prob[testIndex]
    
    return auroc_val, auprc_val, full_prob, test_prob


def trainPred(
    geneList: list[str],
    multi_feature,
    C2_data,
    C5_data,
    positiveGenePath: str,
    negativeGenePath: str,
    lr: float,
    epochs: int,
    dropout: float,
    n_hid: int,
    weight_decay: float,
    base_seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_feature = torch.as_tensor(multi_feature, dtype=torch.float32, device=device)
    multi_feature_np = np.asarray(multi_feature.detach().cpu())

    aurocList = []
    auprcList = []
    predictionRes_full = pd.DataFrame(index=geneList)
    evaluationRes_oof = pd.DataFrame(index=geneList)
    oof_sum = pd.Series(0.0, index=geneList, dtype=np.float64)
    oof_count = pd.Series(0, index=geneList, dtype=np.int64)
    
    for seed_offset in range(5):
        run_seed = base_seed + seed_offset
        set_all_seeds(run_seed)
    
        sampleIndex, label, labelFrame = getData(positiveGenePath, negativeGenePath, geneList)
        sk_X = sampleIndex.reshape([-1, 1])
        sfolder = StratifiedKFold(n_splits=5, random_state=run_seed, shuffle=True)
    
        for fold_id, (train_index, test_index) in enumerate(sfolder.split(sk_X, label), start=1):
            train_global = sampleIndex[train_index]
            test_global = sampleIndex[test_index]
    
            clf = RandomForestClassifier(
                n_estimators=200,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=0,
                max_depth=5,
            )
            clf.fit(multi_feature_np[train_global], label[train_index])
    
            onehot = OneHotEncoder(handle_unknown="ignore")
            onehot.fit(clf.apply(multi_feature_np[train_global]))
            feature_transform = onehot.transform(clf.apply(multi_feature_np))
            feature_transform = torch.from_numpy(feature_transform.toarray()).float().to(device)
    
            auroc_val, auprc_val, full_prob, test_prob = train_test(
                train_global,
                test_global,
                labelFrame,
                feature_transform,
                C2_data,
                C5_data,
                geneList,
                lr,
                epochs,
                dropout,
                n_hid,
                weight_decay,
            )
    
            aurocList.append(float(auroc_val))
            auprcList.append(float(auprc_val))
    
            col_name = f"seed{run_seed}_fold{fold_id}"
            predictionRes_full[col_name] = pd.Series(full_prob, index=geneList, dtype=np.float64)
    
            fold_col = pd.Series(np.nan, index=geneList, dtype=np.float64)
            fold_col.iloc[test_global] = test_prob
            evaluationRes_oof[col_name] = fold_col
    
            tested_genes = pd.Index(np.asarray(geneList)[test_global])
            oof_sum.loc[tested_genes] += test_prob
            oof_count.loc[tested_genes] += 1
    
    prediction_cols = predictionRes_full.columns.tolist()
    predictionRes_full["ensemble_mean"] = predictionRes_full[prediction_cols].mean(axis=1, skipna=True)
    predictionRes_full["ensemble_std"] = predictionRes_full[prediction_cols].std(axis=1, skipna=True)
    
    evaluationRes_oof["oof_count"] = oof_count
    evaluationRes_oof["oof_mean"] = oof_sum / oof_count.replace(0, np.nan)
    
    final_scores = predictionRes_full[["ensemble_mean", "ensemble_std"]].copy()
    final_scores["oof_count"] = oof_count
    final_scores["oof_mean"] = evaluationRes_oof["oof_mean"]
    final_scores["final_score"] = final_scores["oof_mean"].where(
        final_scores["oof_count"] > 0,
        final_scores["ensemble_mean"],
    )
    
    return aurocList, auprcList, predictionRes_full, evaluationRes_oof, final_scores
