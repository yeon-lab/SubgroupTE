# SubgroupTE: Advancing Treatment Effect Estimation with Subgroup Identification

## Introduction
This repository contains source code for paper "SubgroupTE: Advancing Treatment Effect Estimation with Subgroup Identification".

Precise estimation of treatment effects is crucial for evaluating intervention effectiveness. While deep learning models have shown promising performance in treatment effect estimation (TEE), most of them ignore the heterogeneity in treatment effects across subgroups with diverse characteristics, thereby limiting their ability to provide accurate estimation and treatment recommendations for certain groups. In this paper, we propose a new neural network-based framework named SubgroupTE that incorporates subgroup identification and treatment effect estimation. SubgroupTE simultaneously identifies heterogeneous subgroups and estimates treatment effects by subgroup, providing a comprehensive approach to more precisely estimate treatment effects by considering the heterogeneity of responses in the estimation process. In addition, SubgroupTE iteratively optimizes subgrouping and treatment effect estimation, resulting in more accurate subgroup identification and treatment effect estimation. Comprehensive experiments on the synthetic and semi-synthetic datasets exhibit the outstanding performance of SubgroupTE compared with the state-of-the-art models on treatment effect estimation. Additionally, experiments conducted on a real-world opioid use disorder (OUD) dataset demonstrate the potential of our approach to enhance personalized treatment recommendations for OUD patients by not only estimating treatment effects but also identifying heterogeneous subgroups based on patients' medical history. 

## Overview
![figure1](https://github.com/ICDM2023-SubgroupTE/SubgroupTE/assets/54523717/a5723196-306c-4a93-b02d-c842fae935d1)

Figure 1: Architecture of SubgroupTE. The feature representation network transforms the input data into latent feature representations. The subgrouping network pre-estimates the treatment effect and assigns subgroup probabilities. The subgroup-informed prediction network combines the subgroup probability vector and the latent features to estimate treatment effects.


## Installation
Our model depends on Numpy, and PyTorch (CUDA toolkit if use GPU). You must have them installed before using our model
>
* Python 3.9
* Pytorch 1.10.2
* Numpy 1.21.2
* Pandas 1.4.1

This command will install all the required libraries and their specified versions.
```python 
pip install -r requirements.txt
```

## Data preparation
### Synthetic datasets
The downloadable version of the synthetic dataset used in the paper can be accesse in the 'data' folder. 

The structure of the synthetic data:
```
synthetic (dict)     
    |-- 'X': [...]   
    |-- 'T': [...]  
    |-- 'Y': [...]  
    |-- 'TE': [...]  
```
_Note: The simulation for the synthetic dataset is already integrated within 'train.py' file._


### OUD dataset
Please be informed that the OUD dataset utilized in this study is derived from MarketScan claims data. To obtain access to the data, interested parties are advised to contact IBM through [link](https://www.ibm.com/watson-health/merative-divestiture).

## Training and test
### Python command
For training and evaluating the model, run the following code
```python 
# Note 1: hyper-parameters are included in config/*.json.
# Note 2: the code simulates the data.
python train.py --config 'SubgroupTE.json' --data 'Synthetic'
```
  
### Parameters
Hyper-parameters are set in train.py
>
* `data`: dataset to use; {'Synthetic', 'IHDP'}.
* `config`: json file

Hyper-parameters are set in config/*.json
>
* `n_samples`: the number of simulated samples (for the synthetic dataset only)
* `train_ratio`: the ratio of training set
* 'test_ratio`: the ratio of test set
* `n_clusters`: the number of subgroups to identify.
* `emb_dim`: the hidden dimension of the feature representation network.
* `out_dim`: the hidden dimension of the subgroup-informed prediction network.
* `n_layers`: the number of layers in TransformerEncoder
* `init`: method for initializing cluster centroids; {'kmeans++', 'random'}
* `alpha, gamma, and beta`: weights to control losses.
* `metrics`: metrics to print out. It is a list format. Functions for all metrics should be included in 'model/metric.py'.
* `early_stop`: the number of epochs for early stopping
* `monitor`: the criterion for early stopping. The first word is 'min' or 'max', the second one is metric


## Results
#### Treatment effect estimation

Table 1. Comparison of prediction performance on the synthetic and semi-synthetic datasets. The average score and standard deviation under 30 trials are reported.
<p align="center">
<img width="600" alt="image" src="https://github.com/ICDM2023-SubgroupTE/SubgroupTE/assets/54523717/03eb0488-8509-4219-a55f-4837730bd1ee" style="display: inline-block;">
</p>

#### Subgroup identification
<p align="center">
<img width="450" alt="image" src="https://github.com/ICDM2023-SubgroupTE/SubgroupTE/assets/54523717/3f008dc7-e872-4f03-b872-5a0dfaa11499" style="display: inline-block;">
</p>
    
Figure 2. Visualization of boxplots of the treatment effect distribution for the identified subgroups on the test set. Each box represents the interquartile range, spanning from the 25th to the 75th percentiles of the treatment effect. The whiskers represent the range between the 5th and 95th percentiles.


#### Sensitivity analysis 

<p align="center">
<img width="500" alt="image" src="https://github.com/ICDM2023-SubgroupTE/SubgroupTE/assets/54523717/39c52f35-83fd-4b61-9522-9745e415c0e8" style="display: inline-block;">
</p>
Figure 3. Sensitivity analysis conducted for (a) Coefficient and (b) Number of subgroups on the semi-synthetic dataset. For (a), the performance of each coefficient is evaluated while fixing the remaining two coefficients at 1.


#### Complexity analaysis

The complexity analysis involves determining the training time and the number of parameters for each model with their respective optimal hyperparameters on the synthetic dataset. 

Early stopping is applied to all models, where training stops if the loss on validation data does not decrease for 20 epochs. The number of epochs represents training epochs.

<p align="center">
<img width="550" alt="image" src="https://github.com/ICDM2023-SubgroupTE/SubgroupTE/assets/54523717/8ac271c8-3b97-42fd-bde1-0bb8593064a2">
</p>

_* Experiments were conducted using a computing cluster consisting of 42 nodes, each equipped with dual Intel Xeon 8268 processors, 384GB RAM, and dual NVIDIA Volta V100 GPUs with 32GB memory._

_*Pre-trained SubgroupTE on the synthetic dataset is provided in the 'saved' folder._





