import pandas as pd
import random
import os
os.chdir('/Users/apple/Desktop/Study/Traffic/test2')
def select_rows_to_txt(input_csv, output_txt, selected_rows):
    df = pd.read_csv(input_csv, skiprows=1, header=None)  # 跳过标题行

    # 选取指定行
    selected_data = df.iloc[selected_rows]

    with open(output_txt, 'w') as outfile:
        outfile.write("C101\n\nVEHICLE\n")  # 第一行"C101"，第二行空，第三行"VEHICLE"
        
        # 写入标题行并对齐
        for col_name in header:
            outfile.write(str(col_name).center(10))
            outfile.write('\t')
        outfile.write('DEMAND'.center(10))  # 添加DEMAND标题
        outfile.write('\n')
        demand=[]
        # 写入数据并对齐
        for index, row in selected_data.iterrows():
            # 对第一列数据进行特殊处理，确保它们是整数形式
            outfile.write(str(int(row[0])).center(10))
            outfile.write('\t')
            # 写入剩余列的数据并对齐
            for value in row[1:]:
                outfile.write(str(value).center(10))
                outfile.write('\t')
            # 写入随机的DEMAND值
            demand_value = 10*random.randint(1, 3)
            demand.append(demand_value)
            outfile.write(str(demand_value).center(10))
            outfile.write('\n')
        print (demand)
# 示例用法
input_csv = 'wangjing-newloc-simplify.csv'  # 输入CSV文件名
output_txt = 'output.txt'  # 输出文本文件名
selected_rows = [30, 15, 24, 43, 60, 55, 57, 61, 7, 70, 28, 23, 68, 22, 26, 51, 0, 20, 25, 19, 33, 34]  # 要选择的行的索引列表
demand= [10, 20, 30, 20, 30, 30, 30, 30, 20, 30, 30, 30, 30, 20, 10, 30, 20, 30, 10, 30, 30, 30]
header = ["CUST NO.", "XCOORD.", "YCOORD."]  # 标题行

select_rows_to_txt(input_csv, output_txt, selected_rows)
