This directory contains the clustering result file for natural products, `drug_cluster_0.3.pkl`. The file is in Python dictionary format, structured as `'CHEMBL100259': 822`, where the key represents the ChEMBL ID (`chembl_id`), and the value indicates the cluster to which it belongs.

In addition, there are three files: `drug_ECFP.pkl`, `drug_PubChem.pkl`, and `drug_ASP.pkl`. These files store the features of the natural products in our dataset. The format is a Python dictionary, where the key represents the ChEMBL ID (`chembl_id`), and the value contains the feature values.





# NatProNet

## Overview

The project uses machine learning and deep learning to predict the interactions between natural products and proteins.

## Environment and Data

```shell
python = 3.8
PyTorch <= 1.8.1
torchVision = 0.9.1
XGBoost = 2.1.3
pytorch geometric = 1.6.3
torch-Scatter = 2.0.7
torch-Sparse = 0.6.10
scikit-learn = 1.3.2
RDKit = 2024.3.2
fair-esm
```

Unzip the archive in the root directory and copy the three files from the `drug_feature` folder to `./model_and_dataset1/data/nps_feature/` and `./model_and_dataset2/data/nps_feature/`. Copy the three files from the `protein_feature` folder to `./model_and_dataset1/data/protein_feature/` and `./model_and_dataset2/data/protein_feature/`.

The architectures of the **model_and_dataset1** and **model_and_dataset2** files are identical, with the only difference being the dataset. Therefore, the following introduction uses **model_and_dataset1** as an example.

## Deep Learning Task (NatProNet)

1. Navigate to the **./model_and_dataset1/DL/** directory and run:

```shell
python train_test_split_DL.py
```

This splits the dataset based on the clustering results of natural products.

2. Run:

```shell
python data_prepare.py
```

This initializes the data, including the features of natural products and proteins.

3. Run:

```shell
python main.py
```

This begins the training process.

## Machine Learning Task (NatProNet)

进入./model_and_dataset1/ML/目录，执行:

```shell
# Splits the dataset based on the clustering results of natural products.
python train_test_split_ML.py

# Select features and models for training
python main.py -nps_feature ECFP -proteins_feature DPC -model RF

# Batch training
python run.py
```



