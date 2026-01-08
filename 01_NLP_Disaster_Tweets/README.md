# Disaster Tweets Classification: Reaching Top 15% with DistilBERT 🚀

> **Kaggle Competition:** [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)
> **Score:** 0.83450 (Top 15%)
> **Model:** DistilBERT (Hugging Face Transformers)

## 1. Project Overview (项目简介)
在这个项目中，我参加了 Kaggle 的经典的 NLP 入门竞赛：**Real Disaster Tweets**。任务是构建一个机器学习模型，判断一条 Twitter 推文是否在描述真实的灾难（Binary Classification）。

* **难点：** 推文是非正式文本，包含大量的拼写错误、缩写、Emoji 和 URL，且数据集中存在标签噪声。
* **我的方案：** 使用预训练的 **DistilBERT** 模型进行微调 (Fine-tuning)，相比传统 LSTM/RNN 方法，能更好地理解上下文语义。

## 2. Tech Stack (技术栈)
* **Python 3.10**
* **PyTorch** (Deep Learning Framework)
* **Hugging Face Transformers** (Pre-trained Models)
* **Pandas & Scikit-Learn** (Data Analysis)
* **Kaggle GPU (T4 x2)** (Hardware Accelerator)

## 3. My Approach (核心思路)

### 3.1 Data Preprocessing
* 使用 `distilbert-base-uncased` 的 Tokenizer 进行分词。
* 处理缺失值：将 text 字段中的 `NaN` 填充为 "None"。
* 设定 `max_length=128` 以覆盖绝大多数推文长度。

### 3.2 Model Training
我选择了 **DistilBERT**，因为它在保持 BERT 97% 性能的同时，参数量减少了 40%，训练速度提升了 60%。

**Hyperparameters:**
* `batch_size`: 16
* `learning_rate`: 2e-5
* `epochs`: 2 (To prevent overfitting)
* `optimizer`: AdamW

```python
# Training Arguments Configuration
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    eval_strategy="epoch",  # Evaluate every epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    report_to="none"
)
