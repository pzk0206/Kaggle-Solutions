# 🏠 Kaggle 实战：基于 XGBoost 的房价预测全流程

> **项目背景**：基于 Kaggle 经典的 [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 竞赛数据。
> **核心技术**：`Pandas` 清洗、`Log` 平滑变换、`XGBoost` 回归、特征重要性分析。

## 📋 目录
1. [Step 1: 环境准备](#step-1-环境准备)
2. [Step 2: 目标值分析 (Log变换)](#step-2-目标值分析)
3. [Step 3: 缺失值处理](#step-3-缺失值处理)
4. [Step 4: 模型训练](#step-4-模型训练)
5. [Step 5: 结果提交](#step-5-结果提交)

---

## Step 1: 环境准备 <a name="step-1-环境准备"></a>

首先导入必要的库，并读取数据。我们需要把 ID 列单独拿出来，因为它不参与训练。

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

print("✅ 数据读取完成！")
