from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
import torch
import random
import scipy.sparse as sp
from scipy.sparse import coo_matrix
def cal_auc(output, labels):
    outputTest = output.cpu().detach().numpy()
    outputTest = np.exp(outputTest)
    outputTest = outputTest[:,1]
    labelsTest = labels.cpu().numpy()
    AUROC = roc_auc_score(labelsTest, outputTest)
    precision, recall, _thresholds = precision_recall_curve(labelsTest, outputTest)
    AUPRC = auc(recall, precision)
    return AUROC,AUPRC

def getData(positiveGenePath, negativeGenePath, geneList):
    positiveGene = pd.read_csv(positiveGenePath, header = None)
    positiveGene = list(positiveGene[0].values)
    positiveGene = list(set(geneList)&set(positiveGene))
    positiveGene.sort()
    negativeGene = pd.read_csv(negativeGenePath, header = None)     
    negativeGene = negativeGene[0]
    negativeGene = list(set(negativeGene)&set(geneList))
    negativeGene.sort()
    
    #print("positiveGene = ",len(positiveGene))
    labelFrame = pd.DataFrame(data = [0]*len(geneList), index = geneList)
    labelFrame.loc[positiveGene,:] = 1
    positiveIndex = np.where(labelFrame == 1)[0]
    labelFrame.loc[negativeGene,:] = -1
    negativeIndex = np.where(labelFrame == -1)[0]
    labelFrame = pd.DataFrame(data = [0]*len(geneList), index = geneList)
    labelFrame.loc[positiveGene,:] = 1
    
    positiveIndex = list(positiveIndex)
    negativeIndex = list(negativeIndex)
    sampleIndex = positiveIndex + negativeIndex
    sampleIndex = np.array(sampleIndex)
    label = pd.DataFrame(data = [1]*len(positiveIndex) + [0]*len(negativeIndex))
    label = label.values.ravel()
    return  sampleIndex, label, labelFrame

def getHyperGraph(hypergraph_edgesPath, edgesWeightsPath, Genes, used_gene):
    hypergraph_edges = pd.read_csv(hypergraph_edgesPath,index_col=None,header=None)
    hypergraph_edges = hypergraph_edges.values.T
    hypergraph_edges = torch.from_numpy(hypergraph_edges)

    edgesWeights = pd.read_csv(edgesWeightsPath,index_col=None,header=None)
    edgesWeights = edgesWeights.values.T
    edgesWeights = edgesWeights[0]
    edgesWeights = torch.from_numpy(edgesWeights)
    edgesWeights[torch.where(edgesWeights==2)[0]]=1
    edgesWeights = edgesWeights.unsqueeze(1)
    edgesWeights = edgesWeights.float()

    row  = hypergraph_edges[0]
    col  = hypergraph_edges[1]
    data = edgesWeights.squeeze(1)
    coo = coo_matrix((data, (row, col)), shape=(hypergraph_edges[0].max()+1,hypergraph_edges[1].max()+1))
    weighted_Hypergraph = coo.toarray()

    coo = coo_matrix((np.ones_like(edgesWeights.squeeze(1)), (row, col)), shape=(hypergraph_edges[0].max()+1, hypergraph_edges[1].max()+1))
    nonWeighted_Hypergraph = coo.toarray()

    weighted_Hypergraph_frame = pd.DataFrame(data = weighted_Hypergraph, index = Genes)
    nonWeighted_Hypergraph_frame = pd.DataFrame(data = nonWeighted_Hypergraph, index = Genes)
    return weighted_Hypergraph_frame, nonWeighted_Hypergraph_frame

def processingHypergraph():
    C2_geneList = pd.read_csv(r'./Data/fullC2_genes.csv',header=None,index_col=None)
    C2_Genes = list(C2_geneList.iloc[:,0].values)

    C5_geneList = pd.read_csv(r'./Data/fullC5_genes.csv',header=None,index_col=None)
    C5_Genes = list(C5_geneList.iloc[:,0].values)

    used_gene = pd.read_csv(r'./Data/geneList.csv',header=None,index_col=None)
    used_gene = list(used_gene.iloc[:,0].values)
    
    C2_hypergraph_edgesPath = r'./Data/C2_hypergraph.csv'
    C2_edgesWeightsPath = r'./Data/C2_weights.csv'
    C2_weighted_Hypergraph_frame, C2_nonWeighted_Hypergraph_frame = getHyperGraph(C2_hypergraph_edgesPath, C2_edgesWeightsPath, C2_Genes, used_gene)

    C5_hypergraph_edgesPath = r'./Data/C5_hypergraph.csv'
    C5_edgesWeightsPath = r'./Data/C5_weights.csv'
    C5_weighted_Hypergraph_frame, C5_nonWeighted_Hypergraph_frame = getHyperGraph(C5_hypergraph_edgesPath, C5_edgesWeightsPath, C5_Genes, used_gene)

    nonWeighted_marix = pd.concat([C2_nonWeighted_Hypergraph_frame,C5_nonWeighted_Hypergraph_frame],axis=1)
    nonWeighted_marix.columns = np.arange(nonWeighted_marix.shape[1])
    nonWeighted_marix = nonWeighted_marix.loc[used_gene]
    nonWeighted_marix = nonWeighted_marix.fillna(0)

    Weighted_marix = pd.concat([C2_weighted_Hypergraph_frame,C5_weighted_Hypergraph_frame],axis=1)
    Weighted_marix.columns = np.arange(Weighted_marix.shape[1])
    Weighted_marix = Weighted_marix.loc[used_gene]
    Weighted_marix = Weighted_marix.fillna(0)

    C2_nonWeighted_Hypergraph_frame = nonWeighted_marix.iloc[:,:C2_nonWeighted_Hypergraph_frame.shape[1]]
    C5_nonWeighted_Hypergraph_frame = nonWeighted_marix.iloc[:,C2_nonWeighted_Hypergraph_frame.shape[1]:]
    C2_nonWeighted_Hypergraph_frame_values = C2_nonWeighted_Hypergraph_frame.values
    C2_weighted_Hypergraph_frame = Weighted_marix.iloc[:,:C2_nonWeighted_Hypergraph_frame.shape[1]]
    C5_weighted_Hypergraph_frame = Weighted_marix.iloc[:,C2_nonWeighted_Hypergraph_frame.shape[1]:]
    C5_weighted_Hypergraph_frame_values = C5_weighted_Hypergraph_frame.values
    
    C2_hypergraph_edges = torch.nonzero(torch.from_numpy(C2_nonWeighted_Hypergraph_frame.values))
    C2_hypergraph_edges = C2_hypergraph_edges.T
    #print(C2_hypergraph_edges.shape)
    C2_edgesWeights = C2_nonWeighted_Hypergraph_frame_values[C2_hypergraph_edges[0], C2_hypergraph_edges[1]]
    C2_edgesWeights = torch.from_numpy(C2_edgesWeights)
    C2_edgesWeights = C2_edgesWeights.float()
    C2_hypergraph_edges = C2_hypergraph_edges.cuda()
    C2_edgesWeights = C2_edgesWeights.cuda()
    
    C5_hypergraph_edges = torch.nonzero(torch.from_numpy(C5_nonWeighted_Hypergraph_frame.values))
    C5_hypergraph_edges = C5_hypergraph_edges.T
    #print(C5_hypergraph_edges.shape)
    C5_edgesWeights = C5_weighted_Hypergraph_frame_values[C5_hypergraph_edges[0], C5_hypergraph_edges[1]]
    C5_edgesWeights = torch.from_numpy(C5_edgesWeights)
    C5_edgesWeights = C5_edgesWeights.float()
    C5_hypergraph_edges = C5_hypergraph_edges.cuda()
    C5_edgesWeights = C5_edgesWeights.cuda()
    
    C2_data = (C2_nonWeighted_Hypergraph_frame, C2_hypergraph_edges, C2_edgesWeights)
    C5_data = (C5_nonWeighted_Hypergraph_frame, C5_hypergraph_edges, C5_edgesWeights)
    return C2_data, C5_data