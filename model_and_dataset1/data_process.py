# 奖数据集转换成深度学习模型需要的数据集格式

import pandas as pd

# 读取 CSV 文件，指定不含表头（header=None）
df = pd.read_csv('data/nps_protein_interactions_median_np_6.csv', header=None)
cutoff_value = 5

# 选择第 2、4、5 列
selected_columns = df.loc[:, [1, 3, 4]]  # 用 .loc 明确选择列

# 确保第四列是数值类型
selected_columns.loc[:, 4] = pd.to_numeric(selected_columns[4], errors='coerce')

# 处理第四列：小于6为0，大于等于6为1
selected_columns.loc[:, 4] = selected_columns[4].apply(lambda x: 0 if x < cutoff_value else 1)

# 保存为 txt 文件，以空格分隔
selected_columns.to_csv('data/npi_likeness06_cutoff_{}.txt'.format(cutoff_value), sep=' ', index=False, header=False)

print("数据已保存到文件中。")
