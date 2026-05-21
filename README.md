# 基于复杂网络的房颤早期预警模型设计 - 课程报告配套程序

本项目为2026年研究生《高级机器学习理论》课程报告配套代码。项目实现了基于定量递归分析（RQA）与递归复杂网络（RCN）的房颤早期预警分类，主要采用 XGBoost 算法进行二分类预测。

## 1. 运行环境配置 (Environment Setup)

为确保代码能够顺利运行，请在 Python 3.9+ 环境下配置相关依赖。本项目提供了完整的依赖列表。

### 推荐安装方式（使用 pip）：
请在项目根目录下打开终端，运行以下命令安装所有依赖：
```bash
pip install -r requirements.txt

```

**核心依赖库版本参考：**

* numpy == 1.24.3
* pandas == 2.0.3
* wfdb == 4.1.2 (用于读取心电信号)
* neurokit2 == 0.2.5
* hrvanalysis == 1.0.4 (用于RR间期提取与异常剔除)
* pyrqa == 0.1.0 (用于定量递归分析)
* networkx == 3.1 (用于复杂网络特征计算)
* xgboost == 2.0.0
* scikit-learn == 1.3.0
* tqdm, matplotlib, scipy

## 2. 数据集准备

本项目使用 **PAF Prediction Challenge Database**。

1. 请确保项目根目录下存在名为 `paf-prediction-challenge-database-1.0.0/` 的文件夹。
2. 该文件夹内应包含原始的 Holter 心电记录文件（如 `p01.dat`, `p01.hea`, `p01.qrs` 等）。

## 3. 核心文件结构

* `RQN.ipynb`: 定量递归分析（RQA）的特征提取与阈值寻优、XGBoost分类模型代码。
* `RCN.ipynb`: 递归复杂网络（RCN）的拓扑特征提取（度、聚类系数、三阶矩）、特征融合与预测代码。
* `pro_data/`: 运行代码后自动生成的文件夹，用于存放预处理后的 RR 间期 numpy 数组。

## 4. 详细运行流程 (How to Run)

评阅老师您可以按照以下顺序直接运行 Jupyter Notebook 获得与报告一致的结果：

### 第一部分：运行定量递归分析预测 (`RQN.ipynb`)

1. 打开 `RQN.ipynb`。
2. 依次运行所有代码块（Run All）。
3. **预期表现**：代码会自动从数据库提取 NSR 和 PreAF 的 RR 间期并保存在 `pro_data` 中，随后进行进度条展示的阈值寻优，最终输出最佳阈值（0.17）下的分类准确率和详细的 Classification Report。

### 第二部分：运行递归复杂网络预测 (`RCN.ipynb`)

1. 打开 `RCN.ipynb`。
2. 依次运行所有代码块（Run All）。
3. **预期表现**：代码会读取 `pro_data` 中的 RR 间期数据，将递归矩阵转化为复杂网络并提取全局拓扑特征，最后输出 XGBoost 的分类准确率（约 0.77）及详细评估报告。

## 5. 预期运行结果说明

代码完全跑通后，终端/输出台打印的分类性能报告（如 Accuracy、Sensitivity/Recall、Specificity）将与课程报告第5章“实验结果”中的表格数据完全一致。代码中关键函数均已添加详尽的中文注释，解释了数据流转与算法逻辑。

