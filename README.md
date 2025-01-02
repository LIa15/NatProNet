# NatProNet

## Overview

The project uses machine learning and deep learning to predict the interactions between natural products and proteins.

## Environment

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

## Deep Learning Task (NatProNet)

The architectures of the **model_and_dataset1** and **model_and_dataset2** files are identical, with the only difference being the dataset. Therefore, the following introduction uses **model_and_dataset1** as an example.

1. Navigate to the **./model_and_dataset1/** directory and run:

```shell
python data_process.py
```

This will convert the raw dataset (dataset1) into the format required for deep learning.

2. Navigate to the **./DL/** directory and run:

```shell
python train_test_split_DL.py
```

This splits the dataset based on the clustering results of natural products.

3. Run:

```shell
python data_prepare.py
```

This initializes the data, including the features of natural products and proteins.

4. Run:

```shell
python main.py
```

This begins the training process.

## Machine Learning Task (NatProNet)

Navigate to the **./model_and_dataset1/ML/** directory and run:

```shell
# Splits the dataset based on the clustering results of natural products.
python train_test_split_ML.py

# Select features and models for training
python main.py -nps_feature ECFP -proteins_feature DPC -model RF

# Batch training
python run.py
```



