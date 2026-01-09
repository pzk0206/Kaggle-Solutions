# 🏠 Kaggle 实战：基于 XGBoost 的房价预测全流程解析

> **项目背景**：基于 Kaggle 经典的 [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 竞赛数据。
> **核心技术**：`Pandas` 清洗、`Log` 平滑变换、`XGBoost` 回归、特征重要性分析。

## 📋 目录
1. [环境准备](#step-1-环境准备)
2. [目标值分析](#step-2-目标值分析)
3. [缺失值处理](#step-3-缺失值处理)
4. [特征工程](#step-4-特征工程)
5. [模型训练与解释](#step-5-模型训练)
6. [结果提交](#step-6-预测与提交)

---

## Step 1: 环境准备

读取数据并分离 ID 列。

```python
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

# 备份并删除 ID
train_ID = train_df['Id']
test_ID = test_df['Id']
train_df.drop("Id", axis=1, inplace=True)
test_df.drop("Id", axis=1, inplace=True)
# Log 变换
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
# One-Hot Encoding
all_data = pd.get_dummies(all_data)
model_xgb = xgb.XGBRegressor(learning_rate=0.05, n_estimators=2200)
model_xgb.fit(X_train, y_train)
log_predictions = model_xgb.predict(X_test)
final_predictions = np.expm1(log_predictions)
# 生成 CSV...
