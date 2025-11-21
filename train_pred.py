import pandas as pd
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import random
from utils import cal_auc, getData
from models import BNHCEncoder, dualChannelArchitecture, Classifier_1
from sklearn.preprocessing import  OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
def train_test(trainIndex, testIndex, labelFrame, multi_feature, C2_data, C5_data, geneList, lr, epochs, dropout, n_hid, weight_decay):
    trainFrame = labelFrame.iloc[trainIndex]
    trainPositiveGene = list(trainFrame.where(trainFrame==1).dropna().index)
    # hyperedge weight
    positiveMatrixSum = C2_data[0].loc[trainPositiveGene].sum()
    selHyperedgeIndex = np.where(positiveMatrixSum>=1)[0]
    weight = positiveMatrixSum.values
    usedColSum = C2_data[0].values.sum(0) + 1e-8
    C2_hyperedgeWeight = weight/usedColSum
    C2_hyperedgeWeight = torch.from_numpy(C2_hyperedgeWeight).float().cuda()
        
    positiveMatrixSum = C5_data[0].loc[trainPositiveGene].sum()
    selHyperedgeIndex = np.where(positiveMatrixSum>=1)[0]
    weight = positiveMatrixSum.values
    usedColSum = C5_data[0].values.sum(0) + 1e-8
    C5_hyperedgeWeight = weight/usedColSum
    C5_hyperedgeWeight = torch.from_numpy(C5_hyperedgeWeight).float().cuda()
    
    model_C2 = BNHCEncoder(in_dim = multi_feature.shape[1], edge_dim = n_hid, node_dim = n_hid, num_layers = 3, dropout = 0.5, n_class=2)
    model_C5 = BNHCEncoder(in_dim = multi_feature.shape[1], edge_dim = n_hid, node_dim = n_hid, num_layers = 3, dropout =  0, n_class=2)
    classifier_C2 = Classifier_1(in_dim = n_hid,out_dim=2)
    classifier_C5 = Classifier_1(in_dim = n_hid,out_dim=2)
    optimizer_C2 = optim.Adam(list(model_C2.parameters())+list(classifier_C2.parameters()), lr=5e-4, weight_decay=0.0001)
    optimizer_C5 = optim.Adam(list(model_C5.parameters())+list(classifier_C5.parameters()), lr=0.0005, weight_decay=5e-4)
    schedular_C2 = optim.lr_scheduler.MultiStepLR(optimizer_C2, milestones=[100,200,300,400], gamma=0.5)
    schedular_C5 = optim.lr_scheduler.MultiStepLR(optimizer_C5, milestones=[100,200,300,400], gamma=0.5)
    model_fusion = dualChannelArchitecture(featureDim = n_hid, dropout = dropout)
    optimizer = optim.AdamW(model_fusion.parameters(), lr=lr, weight_decay=weight_decay)

    labels = torch.from_numpy(labelFrame.values.reshape(-1,))
    if torch.cuda.is_available():
        multi_feature = multi_feature.cuda()
        labels = labels.cuda()
        model_C2.cuda()
        model_C5.cuda()
        model_fusion.cuda()
        classifier_C2.cuda()
        classifier_C5.cuda()
    for epoch in range(epochs):
        model_C2.train() # 先将model置为训练状态
        model_C5.train()
        classifier_C2.train()
        classifier_C5.train()
        optimizer_C2.zero_grad() # 梯度置0
        optimizer_C5.zero_grad()

        output_C2 = classifier_C2(model_C2(multi_feature, C2_data[1], C2_hyperedgeWeight, C2_data[2]))
        output_C5 = classifier_C5(model_C5(multi_feature, C5_data[1], C5_hyperedgeWeight, C5_data[2]))
        loss_train_C2 = F.nll_loss(
            output_C2[trainIndex], labels[trainIndex])  
        loss_train_C5 = F.nll_loss(
            output_C5[trainIndex], labels[trainIndex]) 
        loss_train_C2.backward() # 反向传播求梯度
        loss_train_C5.backward()
        optimizer_C2.step() # 更新参数
        optimizer_C5.step()
        schedular_C2.step()
        schedular_C5.step()

        if(epoch>epochs/2):
            model_fusion.train()
            optimizer.zero_grad()
            output_fusion = model_fusion(model_C2(multi_feature, C2_data[1], C2_hyperedgeWeight, C2_data[2]),
                                         model_C5(multi_feature, C5_data[1], C5_hyperedgeWeight, C5_data[2]))
            loss_train_fusion = F.nll_loss(output_fusion[trainIndex], labels[trainIndex])
            loss_train_fusion.backward()
            optimizer.step()
    model_C2.eval() 
    model_C5.eval()
    model_fusion.eval()
    classifier_C2.eval()
    classifier_C5.eval() 
    with torch.no_grad():
        output = model_fusion(model_C2(multi_feature, C2_data[1], C2_hyperedgeWeight, C2_data[2]),
                              model_C5(multi_feature, C5_data[1], C5_hyperedgeWeight, C5_data[2]))
        loss_test = F.nll_loss(output[testIndex], labels[testIndex])
        AUROC_val, AUPRC_val = cal_auc(output[testIndex], labels[testIndex])
        outputFrame = pd.DataFrame(data = output.exp().cpu().detach().numpy(), index = geneList)
    return AUROC_val, AUPRC_val, outputFrame


def trainPred(geneList, multi_feature, C2_data, C5_data, positiveGenePath,
              negativeGenePath, lr, epochs, dropout, n_hid, weight_decay):
    aurocList = list()
    auprcList = list()
    evaluationRes = pd.DataFrame(index = geneList)
    for i in range(5):
        sampleIndex,label,labelFrame = getData(positiveGenePath, negativeGenePath, geneList)
        sk_X = sampleIndex.reshape([-1,1])
        sfolder = StratifiedKFold(n_splits = 5, random_state = i, shuffle = True)
        for train_index,test_index in sfolder.split(sk_X, label):
            X_train, y_train = multi_feature[train_index], label[train_index]
            clf = RandomForestClassifier(n_estimators=200, n_jobs=-1,class_weight = "balanced_subsample",random_state=0, max_depth = 5)
            clf.fit(X_train,y_train)
            onehot = OneHotEncoder()
            feature_transform = onehot.fit_transform(clf.apply(multi_feature))
            feature_transform = torch.from_numpy(feature_transform.todense()).float()
            
            trainIndex, testIndex = sampleIndex[train_index], sampleIndex[test_index]
            print("trainModel")
            AUROC_val, AUPRC_val, outputFrame = train_test(trainIndex, testIndex, labelFrame, feature_transform, C2_data, C5_data, geneList, lr, epochs, dropout, n_hid, weight_decay)
            print(AUROC_val)
            print(AUPRC_val)
            aurocList.append(AUROC_val)
            auprcList.append(AUPRC_val)
            evaluationRes = pd.concat([evaluationRes,outputFrame[1]], axis = 1)
    return aurocList, auprcList, evaluationRes
