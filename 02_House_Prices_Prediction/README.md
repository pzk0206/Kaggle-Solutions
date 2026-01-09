# 🏠 Kaggle 实战：基于 XGBoost 的房价预测

> **项目背景**：基于 Kaggle 经典的房价预测比赛。
> **核心技术**：`Pandas` 数据清洗、`Log` 平滑变换、`XGBoost` 回归。

## 📋 目录
1. [Step 1: 环境准备](#step-1)
2. [Step 2: 目标值处理](#step-2)
3. [Step 3: 特征工程](#step-3)
4. [Step 4: 模型训练](#step-4)

---

## Step 1: 环境准备 <a name="step-1"></a>

首先导入必要的库，并读取数据。

```python
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb

# 读取数据
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

# 备份并删除 ID
train_ID = train_df['Id']
test_ID = test_df['Id']
train_df.drop("Id", axis=1, inplace=True)
test_df.drop("Id", axis=1, inplace=True)

print("Step 1 完成")
# Log 平滑变换
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
print("Step 2 完成：目标值已做 Log 变换")
