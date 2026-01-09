# 🏠 Kaggle 实战：基于 XGBoost 的房价预测

> **项目背景**：基于 Kaggle 经典的房价预测比赛。
> **核心技术**：`Pandas` 数据清洗、`Log` 平滑变换、`XGBoost` 回归。

## 📋 目录
1. [Step 1: 环境准备](#step-1)
2. [Step 2: 目标值处理](#step-2)
3. [Step 3: 特征工程](#step-3)
4. [Step 4: 模型训练](#step-4)
5. [Step 5: 结果提交](#step-5)

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

print("Step 1 完成：数据读取成功")
```

---

## Step 2: 目标值处理 <a name="step-2"></a>

原始房价呈现**右偏分布**，我们使用 `log1p` 对其进行平滑处理。

![Target Distribution](images/target_dist.png)

```python
# Log 平滑变换
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
print("Step 2 完成：目标值已做 Log 变换")
```

---

## Step 3: 特征工程 <a name="step-3"></a>

我们对缺失值进行填充（例如没有游泳池填 "None"）。

```python
# 合并数据
ntrain = train_df.shape[0]
y_train = train_df.SalePrice.values
all_data = pd.concat((train_df.drop(["SalePrice"], axis=1), test_df)).reset_index(drop=True)

# 简单填充示例
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data = pd.get_dummies(all_data)

# 重新拆分
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]
print(f"Step 3 完成：特征处理完毕，维度: {all_data.shape}")
```

---

## Step 4: 模型训练 <a name="step-4"></a>

使用 XGBoost 回归模型进行训练。

![Feature Importance](images/feature_importance.png)

```python
# 建立模型
model_xgb = xgb.XGBRegressor(
    learning_rate=0.05, 
    n_estimators=2200,
    max_depth=3
)

# 训练
model_xgb.fit(X_train, y_train)
print("Step 4 完成：训练结束！")
```

---

## Step 5: 结果提交 <a name="step-5"></a>

将预测结果还原（从 Log 变回正常价格）。

```python
log_predictions = model_xgb.predict(X_test)
final_predictions = np.expm1(log_predictions)

# 生成提交文件
submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = final_predictions
submission.to_csv('submission.csv', index=False)
print("Step 5 完成：文件已生成")
```
