# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 01:55:36 2023

@author: starm
"""

import matplotlib.pyplot as plt
import numpy as np

# 模拟一些数据
variables = ["MAPE(%)", "MAE", "RMSE"]
models = ["STGCN-TCN", "STGCN", "ARIMA", "DCRNN"]

# 每个变量的模型性能数据（示例数据，你需要替换成你的实际数据）
data = {
    "MAPE(%)": [5.23, 5.25, 26.36, 5.44],
    "MAE": [2.23, 2.25, 9.14, 2.36],
    "RMSE": [4.07, 4.07, 13.09, 4.42],
}

# 设置柱状图的宽度
bar_width = 0.2

# 生成柱状图
fig, ax = plt.subplots()

for i, model in enumerate(models):
    # 计算每个柱状图的位置
    x = np.arange(len(variables)) + i * bar_width

    # 绘制柱状图
    ax.bar(x, [data[var][i] for var in variables], width=bar_width, label=model)

# 设置X轴标签
ax.set_xticks(np.arange(len(variables)) + (len(models) - 1) * bar_width / 2)
ax.set_xticklabels(variables)

# 设置图例
ax.legend()

# 设置标题和标签
plt.title("Comparison of models on the same testset for each metric (15 minutes)")
plt.xlabel("Metrics")
plt.ylabel("Performances")
###################################################################################
# 模拟一些数据
variables = ["MAPE(%)", "MAE", "RMSE"]
models = ["STGCN-TCN", "STGCN", "ARIMA", "DCRNN"]

# 每个变量的模型性能数据（示例数据，你需要替换成你的实际数据）
data = {
    "MAPE(%)": [7.36, 7.38, 25.80, 8.06],
    "MAE": [3.06, 3.05, 9.05, 3.35],
    "RMSE": [5.82, 5.78, 12.94, 6.38],
}

# 设置柱状图的宽度
bar_width = 0.2

# 生成柱状图
fig, ax = plt.subplots()

for i, model in enumerate(models):
    # 计算每个柱状图的位置
    x = np.arange(len(variables)) + i * bar_width

    # 绘制柱状图
    ax.bar(x, [data[var][i] for var in variables], width=bar_width, label=model)

# 设置X轴标签
ax.set_xticks(np.arange(len(variables)) + (len(models) - 1) * bar_width / 2)
ax.set_xticklabels(variables)

# 设置图例
ax.legend()

# 设置标题和标签
plt.title("Comparison of models on the same testset for each metric (30 minutes)")
plt.xlabel("Metrics")
plt.ylabel("Performances")
##################################################################
# 模拟一些数据
variables = ["MAPE(%)", "MAE", "RMSE"]
models = ["STGCN-TCN", "STGCN", "ARIMA", "DCRNN"]

# 每个变量的模型性能数据（示例数据，你需要替换成你的实际数据）
data = {
    "MAPE(%)": [8.84, 8.87, 25.22, 10.28],
    "MAE": [3.68, 3.62, 8.97, 4.13],
    "RMSE": [7.04, 6.92, 12.79, 7.76],
}

# 设置柱状图的宽度
bar_width = 0.2

# 生成柱状图
fig, ax = plt.subplots()

for i, model in enumerate(models):
    # 计算每个柱状图的位置
    x = np.arange(len(variables)) + i * bar_width

    # 绘制柱状图
    ax.bar(x, [data[var][i] for var in variables], width=bar_width, label=model)

# 设置X轴标签
ax.set_xticks(np.arange(len(variables)) + (len(models) - 1) * bar_width / 2)
ax.set_xticklabels(variables)

# 设置图例
ax.legend()

# 设置标题和标签
plt.title("Comparison of models on the same testset for each metric (45 minutes)")
plt.xlabel("Metrics")
plt.ylabel("Performances")

##################################################### figures for DCRNN MAE MAPE RMSE
data = {
    "15-MAE": [2.41, 2.39, 2.37, 2.37, 2.34, 2.33, 2.40, 2.37, 2.33, 2.36],
    "15-MAPE": [5.67, 5.56, 5.53, 5.48, 5.48, 5.44, 5.60, 5.50, 5.43, 5.44],
    "15-RMSE": [4.50, 4.46, 4.46, 4.47, 4.41, 4.39, 4.42, 4.41, 4.39, 4.42],
    "30-MAE": [3.49, 3.32, 3.30, 3.30, 3.28, 3.25, 3.47, 3.37, 3.29, 3.35],
    "30-MAPE": [8.52, 8.06, 8.09, 7.95, 8.10, 8.06, 8.48, 8.20, 8.03, 8.06],
    "30-RMSE": [6.58, 6.46, 6.39, 6.46, 6.32, 6.28, 6.38, 6.35, 6.31, 6.38],
    "45-MAE": [4.39, 4.05, 4.02, 4.05, 3.99, 3.95, 4.34, 4.17, 4.05, 4.13],
    "45-MAPE": [11.05, 10.12, 10.21, 10.02, 10.28, 10.28, 10.95, 10.50, 10.22, 10.28],
    "45-RMSE": [8.08, 7.89, 7.73, 7.88, 7.63, 7.57, 7.77, 7.71, 7.66, 7.76],
}
import matplotlib.pyplot as plt
import numpy as np


# 创建折线图
fig, ax = plt.subplots()

# 绘制折线图
for feature, values in data.items():
    ax.plot(range(10), values, label=feature)

# 设置图例
legend = ax.legend(
    loc="upper right", bbox_to_anchor=(1.0, 1.0), fancybox=True, framealpha=0.1
)

# 设置标题和标签
plt.xlabel("Epoch index")
plt.ylabel("Values")
plt.tight_layout()
# 展示图形
plt.show()

##################################################### figures for STGCN-original
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = "scripts/original.csv"

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path, header=None)

# Set names for each column (replace with your desired column names)
column_names = [
    "Training_Steps",
    "MAPE-15-V",
    "MAPE-15-T",
    "MAE-15-V",
    "MAE-15-T",
    "RMSE-15-V",
    "RMSE-15-T",
    "MAPE-30-V",
    "MAPE-30-T",
    "MAE-30-V",
    "MAE-30-T",
    "RMSE-30-V",
    "RMSE-30-T",
    "MAPE-45-V",
    "MAPE-45-T",
    "MAE-45-V",
    "MAE-45-T",
    "RMSE-45-V",
    "RMSE-45-T",
]
# Rename the columns
df.columns = column_names

# Plot a line for each column
for column in df.columns:
    plt.figure()  # Create a new figure for each column
    plt.plot(df.index, df[column], label=column)
    plt.title(f"{column}")
    plt.xlabel("Epoch index")
    plt.ylabel("Value")
    plt.legend()

plt.show()
##################################################### figures for STGCN-modified
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = "scripts/modified.csv"

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path, header=None)

# Set names for each column (replace with your desired column names)
column_names = [
    "Training_Steps",
    "MAPE-15-V",
    "MAPE-15-T",
    "MAE-15-V",
    "MAE-15-T",
    "RMSE-15-V",
    "RMSE-15-T",
    "MAPE-30-V",
    "MAPE-30-T",
    "MAE-30-V",
    "MAE-30-T",
    "RMSE-30-V",
    "RMSE-30-T",
    "MAPE-45-V",
    "MAPE-45-T",
    "MAE-45-V",
    "MAE-45-T",
    "RMSE-45-V",
    "RMSE-45-T",
]
# Rename the columns
df.columns = column_names

# Plot a line for each column
for column in df.columns:
    plt.figure()  # Create a new figure for each column
    plt.plot(df.index, df[column], label=column)
    plt.title(f"{column}")
    plt.xlabel("Epoch index")
    plt.ylabel("Value")
    plt.legend()

plt.show()
