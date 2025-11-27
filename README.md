# HyperMI: Integration of Multi-omics Data with Biologically Informed Hypergraph Framework for Cancer Gene Identification
HyperMI is a biologically informed hypergraph multi-omics integration framework to identify cancer genes .
This repo is for the source code of "Integration of Multi-omics Data with Biologically Informed Hypergraph Framework for Cancer Gene Identification". \

Setup
------------------------
The setup process for HyperMI requires the following steps:
### Download
Download HyperMI.  The following command clones the current HyperMI repository from GitHub:

    git clone https://github.com/CharlesDeng0814/HyperMI.git
### Environment Settings
> python==3.6.13 \
> scipy==1.1.0 \
> torch==1.13.0+cu117 \
> numpy==1.15.2 \
> pandas==0.23.4 \
> scikit_learn==0.19.2

GPU: NVIDIA A100 80G\
CPU: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz

### Usage
(1) After downloading and unzipping this repository, go into the folder. 

(2) We have created examples of HyperMI for predicting pan-cancer genes, namely 'main.py'.

Assuming that you are currently in the downloaded folder, just run the following command and you will be able to build a model and make predictions:

predicting pan-cancer genes
```bash
 
python main.py ./outputFile
 
 ```

 ### Output
The output of HyperMI is the ranking results and prediction scores of all genes.

### Files
*main.py*: Examples of HyperMI for cancer gene identification \
*models.py*: HyperMI model \
*train_pred.py*: Training and testing functions \
*utils.py*: Supporting functions

### Cite
```

```

## Contact
If you have any questions, please contact us:<br>
Chao Deng, `deng_chao@csu.edu.cn` <be>
Jianxin Wang, `jxwang@mail.csu.edu.cn` 
