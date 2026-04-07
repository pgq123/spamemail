# 垃圾短信识别实验项目

本项目围绕英文短信垃圾分类任务，提供了一个可复现实验仓库，用于比较三类模型方案：

- 预训练 BERT 微调
- 从零训练的紧凑型 BERT
- 基于 TF-IDF 特征的 MLP 基线模型

项目目标有两个：一是完成一个可直接运行的垃圾短信分类实验；二是为论文撰写提供结构化的实验产物、指标结果和图表文件。

## 项目特点

- 提供统一训练入口，可一条命令跑完三组对比实验
- 同时保留单独训练 BERT 与 MLP 的脚本，便于消融或补充实验
- 自动输出指标、预测结果、训练曲线、混淆矩阵、分类报告等实验材料
- 兼容多种常见数据列命名方式，如 `v1/v2`、`label/text`、`target/text`
- 对 BERT 实验额外记录 `tokenizer_unk_ratio_train`，用于观察 tokenizer 覆盖情况
- 三模型对比支持三种评估方法（绝对值排名/相对最优差距/综合加权评分）并导出统计表
- 对比可视化新增“直方图 + 精确数值表”联动仪表图，便于论文直接引用

## 项目结构

```text
spamemail/
  data/
  docs/
  models/
  results/
  src/
    bert_train.py
    config.py
    metrics.py
    mlp_train.py
    model.py
    preprocess.py
    runtime_utils.py
    train.py
    visualize.py
  tests/
  README.md
  requirements.txt
```

## 数据集说明

当前默认数据文件为：

```text
data/sms_spam_collection.csv
```

项目内部会统一将数据标准化为以下字段：

- `Prediction`：二分类标签，`1` 表示垃圾短信，`0` 表示正常短信
- `text`：短信文本
- `Email No.`：样本编号

若原始数据列名不是上述格式，程序会自动识别以下别名：

- 标签列：`Prediction`、`target`、`label`、`v1`
- 文本列：`text`、`message`、`sms`、`v2`
- 编号列：`Email No.`、`id`

如果输入数据没有原始文本列，而是词频型特征列，预处理模块也会尝试重建伪文本，以兼容 BERT 实验流程。

## 环境依赖

建议使用独立 Python 环境运行。核心依赖包括：

- `torch`
- `transformers`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`

安装依赖：

```powershell
pip install -r requirements.txt
```

## 运行方式

### 1. 一键运行完整对比实验

下面命令会统一运行：

- 预训练 BERT
- 从零训练 BERT
- MLP 基线

```powershell
python -m src.train --project-root . --data-path data/sms_spam_collection.csv
```

运行完成后，会在 `results/bert/`、`results/mlp/` 和 `results/comparison/` 下生成带时间戳的实验目录。

### 2. 单独运行 BERT 实验

默认模式为预训练 BERT 微调：

```powershell
python -m src.bert_train --project-root . --data-path data/sms_spam_collection.csv --epochs 40 --batch-size 16 --learning-rate 3e-5
```

若只使用本地已缓存的预训练模型文件，可加：

```powershell
python -m src.bert_train --project-root . --data-path data/sms_spam_collection.csv --pretrained-local-files-only
```

若需要切换到从零训练的 BERT 主干，可运行：

```powershell
python -m src.bert_train --project-root . --data-path data/sms_spam_collection.csv --from-scratch-backbone --epochs 40 --learning-rate 3e-5
```

### 3. 单独运行 MLP 基线实验

```powershell
python -m src.mlp_train --project-root . --data-path data/sms_spam_collection.csv --learning-rate 0.001 --batch-size 64 --max-iter 40
```

## 实验设置

### 数据划分

三类模型统一采用分层划分，保证对比公平：

- 训练集：70%
- 验证集：15%
- 测试集：15%
- 随机种子：42

### 预训练 BERT 设置

当前仓库中的 BERT 实验默认配置为：

- 预训练主干：`distilbert-base-uncased`
- 最大序列长度：`128`
- 批大小：`16`
- 学习率：`3e-5`
- 权重衰减：`0.01`
- 训练轮数：`40`
- warmup ratio：`0.1`
- 启用类别权重平衡损失

### 从零训练 BERT 设置

从零训练模式不加载外部预训练权重，而是基于当前数据集词表构建 tokenizer，并训练一个紧凑型 BERT 分类器。对应配置包括：

- `hidden_size=128`
- `num_hidden_layers=2`
- `num_attention_heads=4`
- `intermediate_size=256`
- `dropout=0.2`

### MLP 基线设置

MLP 基线采用 TF-IDF + 两层前馈神经网络：

- 最大特征数：`6000`
- n-gram：`(1, 2)`
- 隐藏层：`(256, 128)`
- 激活函数：`ReLU`
- 学习率：`0.001`
- 批大小：`64`
- 最大迭代轮数：`40`
- `alpha=1e-4`

## 当前实验结果

根据仓库中 `results/comparison/24-16-51/three_model_metrics_comparison.json` 的最新对比结果：

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 预训练 BERT | 0.992 | 0.949 | 0.991 | 0.969 | 0.999 |
| MLP | 0.976 | 0.918 | 0.902 | 0.910 | 0.973 |
| 从零训练 BERT | 0.802 | 0.394 | 0.893 | 0.546 | 0.876 |

从当前结果看：

- 预训练 BERT 表现最好，在 Accuracy、F1 和 ROC-AUC 上均排名第一
- MLP 作为轻量基线表现稳定，整体性能较强，适合资源受限场景
- 从零训练 BERT 虽然召回率较高，但精确率明显偏低，说明仅依靠当前数据规模不足以支撑随机初始化 Transformer 获得理想泛化能力

## 输出文件说明

### BERT 结果目录

输出路径格式通常为：

```text
results/bert/<时间戳>/<pretrained|zero_trained>/
```

常见文件包括：

- `bert_*_training_history.csv`
- `bert_*_test_metrics.json`
- `bert_*_classification_report.csv`
- `bert_*_test_predictions.csv`
- `bert_*_split_manifest.json`
- `bert_*_experiment_config.json`
- `bert_*_training_curves.png`
- `bert_*_metric_bars.png`
- `bert_*_confusion_matrix.png`
- `bert_*_attention_heatmap.png`

模型权重输出到：

- `models/<时间戳>/<pretrained|zero_trained>/bert_*_spam_classifier.pt`

若是从零训练 BERT，还会额外生成词表文件：

- `models/<时间戳>/zero_trained/bert_zero_trained_vocab.txt`

### MLP 结果目录

输出路径格式通常为：

```text
results/mlp/<时间戳>/
```

常见文件包括：

- `mlp_test_metrics.json`
- `mlp_training_history.csv`
- `mlp_classification_report.csv`
- `mlp_test_predictions.csv`
- `mlp_confusion_matrix.csv`
- `mlp_confusion_matrix.png`
- `mlp_metric_bars.png`
- `mlp_training_curves.png`
- `mlp_experiment_config.json`

模型文件输出到：

- `models/<时间戳>/mlp_spam_classifier.pkl`

### 三模型对比结果

统一训练入口还会额外生成对比文件：

```text
results/comparison/<时间戳>/
```

包含：

- `three_model_metrics_comparison.csv`
- `three_model_metrics_comparison.json`
- `three_model_performance_methods.csv`
- `three_model_metric_statistics.csv`
- `three_model_metrics_comparison.png`
- `three_model_metrics_dashboard.png`
- `chapter_4_to_5_bridge.png`

其中：

- `three_model_performance_methods.csv`：三种评估方法下每个模型的得分与排序
- `three_model_metric_statistics.csv`：关键指标的均值、标准差、最优/最差模型与数值
- `three_model_metrics_dashboard.png`：柱状图与精确表格联动展示

### 系统流程图产物

流程图可编辑源文件与导出文件位于：

- `docs/flowcharts/system_implementation_flowchart.mmd`（可编辑 Mermaid 源）
- `docs/flowcharts/system_implementation_flowchart.png`（高清位图）
- `docs/flowcharts/system_implementation_flowchart.pdf`（矢量文档）

可通过以下命令重新导出 PNG/PDF：

```powershell
conda run -n paper python -m src.flowchart_export --output-dir docs/flowcharts
```

模型架构图（MLP 与 DistilBERT）可通过以下命令导出：

```powershell
conda run -n paper python -m src.model_architecture_export --output-dir docs/flowcharts
```

核心代码截图可通过以下命令导出：

```powershell
conda run -n paper python -m src.code_snapshot_export --project-root . --output-dir docs/code_snapshots
```

## 测试

运行单元测试：

```powershell
python -m unittest discover -s tests -v
```

测试主要覆盖：

- 数据适配与列名兼容
- 训练入口参数行为
- BERT 与 MLP 关键训练逻辑
- 指标计算与可视化输出

## 适合论文写作的使用方式

如果你需要把这个仓库用于课程论文或毕业论文，比较推荐的写法是：

1. 将 MLP 作为传统文本特征方法基线
2. 将预训练 BERT 作为主模型
3. 将从零训练 BERT 作为“预训练有效性”验证实验
4. 重点分析 Accuracy、Precision、Recall、F1 和 ROC-AUC
5. 结合混淆矩阵与注意力热力图补充可解释性说明

更详细的实验写作说明见：

- `docs/paper_experiment_notes_zh.md`
