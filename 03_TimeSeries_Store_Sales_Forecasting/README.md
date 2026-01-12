# 📈 Store Sales - Time Series Forecasting (Top 20% 实战解析)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Score](https://img.shields.io/badge/Public_LB-0.46139-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 1. 项目背景 (Overview)

本项目是 Kaggle 经典时间序列竞赛 [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) 的完整解决方案。

* **业务场景：** 预测厄瓜多尔大型零售商 Favorita 旗下 54 家商店在未来 16 天的日销量。
* **难点：** 数据受多种外部因素影响（油价波动、地方性假期、发薪日效应），且不同商店、不同品类的销售模式差异巨大。
* **最终成绩：** RMSLE **0.46139** (Top 20%)。

---

## 2. 核心策略与演进 (Strategy)

我不追求盲目的模型堆叠，而是采用**工业级特征工程**的思路，通过“理解数据”来提升分数。

| 阶段 | 方法 | 验证集分数 (RMSLE) | 提升逻辑 |
| :--- | :--- | :--- | :--- |
| **Baseline** | 线性回归 (仅日期特征) | 2.19 | 模型欠拟合，无法捕捉非线性关系。 |
| **V2** | XGBoost + 宏观数据 | 0.69 | 引入油价和商店位置，确立了基础树模型框架。 |
| **V3** | 精细化假期匹配 | 0.68 | 修正了“所有假期都影响所有商店”的错误逻辑。 |
| **V5** | **滞后特征 (Lag Features)** | **0.5064** | **质变点：** 教会模型理解“近期趋势”和“历史惯性”。 |

---

## 3. 技术深度解析 (Technical Deep Dive)

以下是本项目最核心的四个技术环节，包含**代码实现**与**设计原理**。

### 3.1 验证集切分策略 (Time-Based Split)

在时间序列中，随机切分（Random Split）是严重的错误，因为它会导致**未来数据泄露 (Data Leakage)**。我严格按照时间轴进行切分。

* **训练集：** 2013-01-01 至 2016-12-31
* **验证集：** 2017-01-01 至 2017-08-15
* **测试集：** 2017-08-16 至 2017-08-31

```python
# 数据切分逻辑
# log1p: 对销量做 Log 变换，使分布更接近正态，且符合 RMSLE 评估指标
y = np.log1p(train['sales'])

# 严格按日期切分
train_mask = train['date'] < '2017-01-01'
val_mask = train['date'] >= '2017-01-01'

X_train = train.loc[train_mask, features]
X_val = train.loc[val_mask, features]
