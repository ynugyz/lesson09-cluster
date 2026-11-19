# 课程 9 · 聚类客户分群

《人工智能程序设计与实践》第 9 次课堂作业。实验对 200 条教学用虚构银行客户记录做 K-means 分群。

## 复现

在本目录运行，验证环境为 Python 3.14。

```bash
python -m pip install -r requirements.txt
python cluster.py
python research.py
```

- [进度报告](progress.md)给出三个课堂检查点、四个中文业务画像、自主实践和独立研究。
- [客户分群脚本](cluster.py)生成 [簇分配表](results/assignments.csv)、[计算明细](results/metrics.json)、[肘部图](results/elbow.png)、[分群画像图](results/profiles.png)和[散点图](results/scatter.png)。
- [同心圆实验](research.py)对比 K-means 与 DBSCAN，生成[对照图](results/rings.png)。

## 原始数据

[`data/bank_customers.csv`](data/bank_customers.csv)取自教师案例仓库的固定版本，保留原始字节和换行。程序在读取前校验 SHA-256，簇标签写入独立结果文件。数据为教学虚构数据，字段和单位见[教师数据字典](https://github.com/thiswind/ai-programming-2026/blob/e8f8d9bfb7ba30c91727601a74b35f242e3daa9e/data/DATA_DICT.md)。[原始 CSV](https://github.com/thiswind/ai-programming-2026/blob/e8f8d9bfb7ba30c91727601a74b35f242e3daa9e/data/bank_customers.csv)

```text
SHA-256  b82ef3124080469b7f208e246f08b6de66bc2a057e0aa84d6eea481716d578da
Git blob a2668e916be6efa7213881378e1298713da08722
```

基线遵循[教师第 9 课案例](https://github.com/thiswind/ai-programming-2026/blob/e8f8d9bfb7ba30c91727601a74b35f242e3daa9e/lessons/09-clustering/cluster.py)的四个特征、`random_state=42` 和 `n_init=10`。本地课堂演示包《lesson09-课堂演示包》中的商场客户实验用于理解量纲对比和两种选 k 依据，作业计算使用上面的银行数据。
