import pandas as pd
import sys, os, random
import numpy as np
import scipy.sparse as sp
from train_pred import trainPred
from utils import processingHypergraph
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":    
    _, outputPath = sys.argv
    lr = 0.0001
    weight_decay = 0.005
    epochs = 200
    n_hid = 256
    dropout = 0
    
    positiveGenePath = r'./Data/796true.txt'
    negativeGenePath = r'./Data/2187false.txt'
    geneList = pd.read_csv(r'./Data/geneList.csv',header=None,index_col=None)
    geneList = list(geneList.iloc[:,0].values)
    multiFeature = pd.read_csv(r'./Data/multiOmicsFeature.csv',index_col = 0)
    multiFeature = multiFeature.loc[geneList].values

    C2_data, C5_data = processingHypergraph()
    aurocList, auprcList, evaluationRes = trainPred(geneList, multiFeature, C2_data, C5_data, positiveGenePath,
                                          negativeGenePath, lr, epochs, dropout, n_hid, weight_decay) 
    predRes = evaluationRes.sum(1).sort_values(ascending = False) / 25
    predRes.to_csv(outputPath,sep='\t', header = False)
    print(np.mean(aurocList)) # 0.948
    print(np.mean(auprcList)) # 0.899