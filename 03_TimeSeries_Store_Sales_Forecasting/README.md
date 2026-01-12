# Store Sales Prediction: Industrial Time Series Forecasting with XGBoost 📈

> **Kaggle 竞赛:** [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
> **公开榜得分:** 0.46139 (Top 20%) 🚀
> **核心模型:** XGBoost Regressor (GPU Accelerated)
> **关键策略:** Lag Features (滞后特征) + Rolling Windows (滑动窗口) + Time-Based Split

## 1. Project Overview (项目简介)
本项目基于 Kaggle 经典的时间序列竞赛。任务是预测厄瓜多尔大型零售商 Corporación Favorita 旗下 **54 家商店**、**33 类商品**在未来 **16 天**的日销量。

* **难点 (Challenges)：**
    * **多变量干扰：** 销量受油价波动（宏观经济）、节假日（局部事件）、发薪日等多重因素影响。
    * **数据量大：** 训练集包含超过 300 万行数据。
    * **未来泄露风险：** 测试集要求预测未来 16 天，必须防止在特征工程中“看见未来”。
* **我的方案 (My Approach)：**
    * **目标变换：** 使用 **Log1p** 处理长尾分布的销量数据。
    * **环境感知：** 构建精准的**假期匹配逻辑**（城市对城市）和油价插值。
    * **时序魔法：** 放弃简单的日期特征，转而构建 **Lag 16+** (滞后特征) 和 **Rolling Mean** (趋势特征)，这是提分的关键。

## 2. Tech Stack (技术栈)
* **Python 3.8+**
* **Pandas & NumPy** (High-performance Data Manipulation)
* **XGBoost** (Gradient Boosting with `tree_method='hist'`)
* **Scikit-Learn** (Label Encoding, Metrics)
* **Matplotlib** (Visualization)

---

## 3. Implementation Details (核心实现)

### 3.1 Data Preprocessing & Context Engineering (数据预处理与环境感知)

为了构建高质量的训练数据，我执行了三个关键步骤：
1.  **目标变换：** 对长尾分布的 `sales` 进行 **Log1p** 变换，使其符合 RMSLE 评估指标。
2.  **环境感知 (Context)：** 编写**精准假期匹配逻辑**。单纯的 Merge 会引入噪音（例如“基多”的商店不应受“昆卡”地方假期的影响），只有当 `Store City == Holiday Locale` 时才标记为假期。
3.  **时间切分 (Split)：** 严禁随机切分，严格按照时间轴划分训练集 (`2013-2016`) 和验证集 (`2017`)。

![Target Distribution](images/target_dist.png)

```python
import pandas as pd
import numpy as np
import xgboost as xgb

# 1. 目标值 Log 平滑 (Target Log Transformation)
train['sales'] = np.log1p(train['sales'])

# 2. 假期特征精准匹配 (Precise Holiday Matching)
# 策略：只有当 商店所在城市 == 假期庆祝城市 时，才标记为假期
def apply_local_holidays(df, local_hols, merge_col):
    merged = df.merge(local_hols[['date', 'locale_name']], 
                      left_on=['date', merge_col], 
                      right_on=['date', 'locale_name'], 
                      how='left')
    is_local_hol = merged['locale_name'].notna()
    # 这是一个累加过程，保留已有的假期标记
    return np.maximum(df.get('is_holiday', 0), is_local_hol.astype(int))

# 初始化并应用逻辑
train['is_holiday'] = 0
train['is_holiday'] = apply_local_holidays(train, local_holidays, 'city')
train['is_holiday'] = apply_local_holidays(train, regional_holidays, 'state')

print("✅ 假期特征清洗完成 (Noise Reduction Applied)")

# 3. 基于时间的严格切分 (Time-Based Split)
# 训练集: 2013 ~ 2016 | 验证集: 2017-01-01 ~ 2017-08-15
train_mask = train['date'] < '2017-01-01'
val_mask = train['date'] >= '2017-01-01'

X_train = train.loc[train_mask, features]
y_train = train.loc[train_mask, 'sales']
X_val = train.loc[val_mask, features]
y_val = train.loc[val_mask, 'sales']

print(f"✅ 数据准备完成。训练集样本数: {X_train.shape[0]}")
