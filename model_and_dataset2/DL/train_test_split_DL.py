# Clustering cross-validation dataset split
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import KFold
from collections import OrderedDict
import os


if __name__ == "__main__":
    # File name
    source_data = "../data/dataset2.csv"
    # Active threshold selection
    cutoff_value = 5
    cluster_data = "../data/nps_feature/drug_cluster_non_weak_0.3.pkl"
    random_seed = 1234
    n_folds = 5


    # Read the CSV file, specifying no header (header=None)
    df = pd.read_csv(source_data, header=None)

    # Select columns 2, 4, and 5
    selected_columns = df.loc[:, [0, 1, 2, 3, 4]]  # Explicitly select columns using .loc

    # Ensure the fourth column is numeric
    selected_columns.loc[:, 4] = pd.to_numeric(selected_columns[4], errors='coerce')

    # Process the fourth column: values less than 6 are 0, greater than or equal to 6 are 1
    selected_columns.loc[:, 4] = selected_columns[4].apply(lambda x: 0 if x < cutoff_value else 1)

    with open(cluster_data, 'rb') as f:
        C_cluster_dict = pickle.load(f)

    C_cluster_ordered = list(OrderedDict.fromkeys(C_cluster_dict.values()))

    C_cluster_list = np.array(C_cluster_ordered)

    # Get all values from the first column and convert to a list
    compound_id = selected_columns.iloc[:, 0].values.tolist()

    # Find corresponding values in C_cluster_dict based on compound_id and create a new column
    selected_columns['cluster_num'] = [C_cluster_dict[compound] for compound in compound_id]

    # n-fold split
    c_kf = KFold(n_folds, shuffle=True, random_state=random_seed)

    c_train_clusters, c_test_clusters = [], []
    fold = 1
    for train_idx, test_idx in c_kf.split(C_cluster_list):  # .split(C_cluster_list):
        df_train =  selected_columns[selected_columns['cluster_num'].isin(train_idx)]
        df_test = selected_columns[selected_columns['cluster_num'].isin(test_idx)]

        df_train_output = df_train.loc[:, [1, 3, 4]]
        df_test_output = df_test.loc[:, [1, 3, 4]]

        # Ensure directories exist
        os.makedirs(os.path.dirname('./dataset/fold_{}/train_data'.format(fold)), exist_ok=True)
        os.makedirs(os.path.dirname('./dataset/fold_{}/test_data'.format(fold)), exist_ok=True)
        df_train_output.to_csv('./dataset/fold_{}/train_data.txt'.format(fold), sep=' ', index=False, header=False)
        df_test_output.to_csv('./dataset/fold_{}/test_data.txt'.format(fold), sep=' ', index=False, header=False)
        fold += 1

    print("end")
