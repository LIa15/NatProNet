# 聚类交叉验证划分数据集
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import KFold
from collections import OrderedDict
import os


if __name__ == "__main__":
    # 文件名称
    source_data = "../data/nps_protein_interactions_median_np_6.csv"
    # 活性阈值选择
    cutoff_value = 5
    cluster_data = "../data/nps_feature/drug_cluster_0.3.pkl"
    random_seed = 1234
    n_folds = 5


    # 读取 CSV 文件，指定不含表头（header=None）
    df = pd.read_csv(source_data, header=None)

    # 选择第 2、4、5 列
    selected_columns = df.loc[:, [0, 1, 2, 3, 4]]  # 用 .loc 明确选择列

    # 确保第四列是数值类型
    selected_columns.loc[:, 4] = pd.to_numeric(selected_columns[4], errors='coerce')

    # 处理第四列：小于6为0，大于等于6为1
    selected_columns.loc[:, 4] = selected_columns[4].apply(lambda x: 0 if x < cutoff_value else 1)

    with open(cluster_data, 'rb') as f:
        C_cluster_dict = pickle.load(f)

    C_cluster_ordered = list(OrderedDict.fromkeys(C_cluster_dict.values()))

    C_cluster_list = np.array(C_cluster_ordered)

    # 获取第一列的所有值并转为列表
    compound_id = selected_columns.iloc[:, 0].values.tolist()

    # 根据 compound_id 在 C_cluster_dict 中查找对应值，生成新列
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

        # 确保目录存在
        os.makedirs(os.path.dirname('./dataset/fold_{}/train_data'.format(fold)), exist_ok=True)
        os.makedirs(os.path.dirname('./dataset/fold_{}/test_data'.format(fold)), exist_ok=True)
        df_train_output.to_csv('./dataset/fold_{}/train_data.txt'.format(fold), sep=' ', index=False, header=False)
        df_test_output.to_csv('./dataset/fold_{}/test_data.txt'.format(fold), sep=' ', index=False, header=False)
        fold += 1

    print("end")
