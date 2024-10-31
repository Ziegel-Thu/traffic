import pandas as pd
import os
os.chdir('/Users/apple/Desktop/Study/Traffic/Test3')
# 读取 list.csv 文件
df_list = pd.read_csv('list.csv', header=None)

# 指定要提取的行和列索引
rows_to_keep = [31, 16, 25, 44, 61, 56, 58, 62, 8, 71, 29, 24, 69, 23, 27, 52, 0, 21, 26, 20, 34, 35]
cols_to_keep = rows_to_keep  # 与行一致，因为要截取方阵

# 提取指定的行和列
extracted_data = df_list.iloc[rows_to_keep, cols_to_keep]

# 将提取的数据保存为 location.csv
extracted_data.to_csv('location.csv', index=False, header=False)

# 打印提示信息
print("提取并保存 location.csv 成功！")
