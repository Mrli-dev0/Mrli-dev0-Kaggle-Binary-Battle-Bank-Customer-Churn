# Binary Battle ML: Bank Customer Churn Challenge

\!\[GitHub license\]\(https://img\.shields\.io/badge/license\-MIT\-blue?style=flat\-square\)

\!\[Python\]\(https://img\.shields\.io/badge/Python\-3\.8%2B\-brightgreen?style=flat\-square\)

\!\[Score\]\(https://img\.shields\.io/badge/Score\-0\.89122\-red?style=flat\-square\)

\!\[Rank\]\(https://img\.shields\.io/badge/Rank\-1st%20%28Top%201%29\-yellow?style=flat\-square\)

\!\[Task\]\(https://img\.shields\.io/badge/Task\-Binary%20Classification\-purple?style=flat\-square\)

# 🏆 银行客户流失预测比赛第一名解决方案

## 📋 比赛概况

- **比赛名称**：Binary Battle ML: Bank Customer Churn Challenge

- **比赛链接**：https://www\.kaggle\.com/competitions/binary\-battle\-ml\-bank\-customer\-churn\-challenge

- **任务类型**：银行客户流失二分类预测

- **评估指标**：AUC

- **最终得分**：0\.89122

- **最终排名**：第 1 名（Top 1）

## 📁 项目结构

```bash
.
├── data/                # 数据集文件夹
│   ├── train.csv        # 训练集
│   ├── test.csv         # 测试集
│   └── sample_submission.csv  # 提交示例
├── images/              # EDA 可视化图表文件夹
├── src/                 # 源代码
│   ├── train.py         # 训练脚本
│   ├── infer.py         # 推理脚本
│   └── utils.py         # 工具函数
├── 01_eda.ipynb         # 数据分析
├── 02_model.ipynb       # 模型训练
├── submission.csv       # 最终提交结果
└── README.md            # 说明文档
```

## 🚀 快速运行

```bash
# 训练模型
python src/train.py

# 生成提交文件
python src/infer.py
```

## ✨ 核心方案思路

### 1\. 探索性数据分析（EDA）

- 分析目标变量分布

- 挖掘关键影响因素：年龄、账户余额、产品数量、活跃度

- 特征相关性与业务逻辑梳理

### 2\. 高阶特征工程

- 交叉特征、比例特征、对数特征构建

- TargetEncoder、WOE 编码、分箱编码

- 业务驱动强区分特征

### 3\. 数据预处理

- KNN 填充缺失值

- 特征标准化

- 互信息法特征选择

### 4\. 多模型融合框架

- 10 折分层交叉验证

- LightGBM \+ XGBoost \+ CatBoost \+ MLP

- Stacking 融合（逻辑回归作为元学习器）

### 5\. 半监督学习优化

- 高质量伪标签自训练

- 提升边界样本预测能力

## 📊 模型与评估

|项目|详情|
|---|---|
|评估指标|AUC|
|验证方式|10 折分层交叉验证|
|模型架构|4基模型 \+ Stacking 融合|
|最终提交|概率预测结果|

## 🛠 环境依赖

```bash
pip install pandas==2.2.3 numpy==2.2.6 dask==2026.3.0
pip install lightgbm==4.6.0.99 xgboost==3.2.0 catboost==1.2.10
pip install scikit-learn==1.8.0 torch==2.6.0+cu124
pip install category_encoders==2.9.0 optuna==4.8.0
pip install matplotlib==3.10.8 seaborn==0.13.2 joblib==1.5.3
```

## 📌 项目说明

本方案以特征工程为核心，结合树模型与深度学习优势，通过严谨的交叉验证与模型融合策略，充分挖掘数据价值，最终在比赛中取得第 1 名。

代码结构清晰、可复现性强，适合同类二分类任务参考。

## 💡 技术栈

\!\[LightGBM\]\(https://img\.shields\.io/badge/LightGBM\-4\.6\.0\.99\-blue?style=flat\-square\)

\!\[XGBoost\]\(https://img\.shields\.io/badge/XGBoost\-3\.2\.0\-green?style=flat\-square\)

\!\[CatBoost\]\(https://img\.shields\.io/badge/CatBoost\-1\.2\.10\-orange?style=flat\-square\)

\!\[PyTorch\]\(https://img\.shields\.io/badge/PyTorch\-2\.6\.0%2Bcu124\-purple?style=flat\-square\)

\!\[Pandas\]\(https://img\.shields\.io/badge/Pandas\-2\.2\.3\-teal?style=flat\-square\)

> （注：文档部分内容可能由 AI 生成）
