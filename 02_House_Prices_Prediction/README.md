# 🏠 Kaggle 实战：基于 XGBoost 的房价预测全流程解析

> **项目背景**：基于 Kaggle 经典的 [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 竞赛数据。
> **核心技术**：`Pandas` 清洗、`Log` 平滑变换、`XGBoost` 回归。

## 📋 目录
1. [Step 1: 环境准备](#step-1-环境准备)
2. [Step 2: 目标值分析 (Log变换)](#step-2-目标值分析)
3. [Step 3: 缺失值处理](#step-3-缺失值处理)
4. [Step 4: 模型训练](#step-4-模型训练)

---

## Step 1: 环境准备

首先导入库，并把 ID 列单独拿出来（因为它不参与训练）。

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
# Log 平滑变换
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
print("✅ 目标值已完成 Log 变换")
# 合并数据以便统一处理
ntrain = train_df.shape[0]
y_train = train_df.SalePrice.values
all_data = pd.concat((train_df.drop(["SalePrice"], axis=1), test_df)).reset_index(drop=True)

# 简单示例：填充缺失值
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["GarageArea"] = all_data["GarageArea"].fillna(0)

# One-Hot 编码 (将文本转为数字)
all_data = pd.get_dummies(all_data)
print(f"特征处理完成，总特征数: {all_data.shape[1]}")

# 重新拆分
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]
# 建立模型
model_xgb = xgb.XGBRegressor(
    learning_rate=0.05, 
    n_estimators=2200,
    max_depth=3
)

# 训练
print("🚀 开始训练...")
model_xgb.fit(X_train, y_train)
print("🎉 训练结束！")
log_predictions = model_xgb.predict(X_test)
final_predictions = np.expm1(log_predictions)

# 生成 CSV
submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = final_predictions
submission.to_csv('submission.csv', index=False)
