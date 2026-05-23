# Medical Agent Pipeline

这是一个多阶段医疗数据处理与建模数据集生成项目。主链路由 `main_orchestrator` 编排，按任务需要调用 Step1 到 Step7 的子 Agent，完成原始医疗数据整理、结构化抽取、任务裁剪、数据清洗、患者一致性验证和 ML 训练数据生成。

本文档面向“第一次拿到项目的人”。重点说明运行前必须配置什么、每一步输入输出在哪里、如何启动、如何续跑，以及常见错误怎么定位。

## 0. 给别人直接照着跑的版本

如果你只是把项目交给别人，让对方先跑起来，优先让他看这一节。

### 0.1 第一次运行前先做这几件事

```bash
cd /Users/mkbk/PycharmProjects/new
conda activate py310

cp configs/model_config.yaml configs/model_config.local.yaml
```

然后打开：

```text
configs/model_config.local.yaml
```

至少填这些模型配置：

```yaml
agents:
  main_orchestrator:
    api_key: "你的 API Key"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4.1-mini"

  agent_1:
    api_key: "你的 API Key"
    base_url: "https://api.minimaxi.com/v1"
    model: "MiniMax-M2.7"

  agent_2_3:
    api_key: "你的 API Key"
    base_url: "https://api.minimaxi.com/v1"
    model: "MiniMax-M2.7"

  agent_4:
    api_key: "你的 API Key"
    base_url: "https://api.minimaxi.com/v1"
    model: "MiniMax-M2.7"

  agent_5:
    api_key: "你的 API Key"
    base_url: "http://127.0.0.1:8317/v1"
    model: "gpt-5.1"

  agent_6_7:
    api_key: "你的 API Key"
    base_url: "https://api.minimaxi.com/v1"
    model: "MiniMax-M2.7"
```

如果对方没有本地 DocLayout-YOLO 权重和 ICD BERT 向量库，先关掉这两个可选能力：

```bash
export STEP1_DOCLAYOUT_ENABLED=0
export ICD10_USE_BERT=0
```

### 0.2 启动交互终端

运行：

```bash
python main_orchestrator/main_orchestrator.py
```

看到：

```text
[MultiAgent] 交互模式已启动。直接输入任务，输入 help 查看示例，输入 exit 退出。

MultiAgent>
```

就在 `MultiAgent>` 后面这样说：

```text
请处理/Users/mkbk/PycharmProjects/new/mimic-mini这个数据集，然后任务裁剪定为肝癌诊断
```

如果你的数据目录不是这个路径，把路径换掉：

```text
请处理<你的数据集绝对路径>这个数据集，然后任务裁剪定为<你的任务目标>
```

例如：

```text
请处理/Users/yourname/project/mimic-mini这个数据集，然后任务裁剪定为死亡预测
```

```text
请处理/Users/yourname/project/rawdata这个数据集，然后任务裁剪定为离院去向预测
```

```text
请处理/Users/yourname/project/mimic-mini这个数据集，然后任务裁剪定为肝癌诊断
```

### 0.3 如果终端追问任务裁剪目标

有时系统会继续问：

```text
[MultiAgent] 请输入本次任务裁剪的目标:
```

这时只输入任务目标本身，不要再输入路径：

```text
肝癌诊断
```

或：

```text
死亡预测
```

或：

```text
离院去向预测
```

### 0.4 不想开交互，直接命令行跑完整流程

如果想一条命令跑完整链路：

```bash
python main_orchestrator/main_orchestrator.py \
  "请处理/Users/mkbk/PycharmProjects/new/mimic-mini这个数据集，然后任务裁剪定为肝癌诊断" \
  --task-type full_pipeline
```

更通用的模板：

```bash
python main_orchestrator/main_orchestrator.py \
  "请处理<你的数据集绝对路径>这个数据集，然后任务裁剪定为<你的任务目标>" \
  --task-type full_pipeline
```

### 0.5 只想先测试 Step1

第一次部署新机器时，建议先只跑 Step1：

```bash
python main_orchestrator/main_orchestrator.py \
  "/Users/mkbk/PycharmProjects/new/mimic-mini" \
  --task-type step1_only \
  --disable-memory-agent \
  --json
```

跑完后检查：

```text
reorganized_output/_meta/records.json
program/output/step1_results/
```

如果这两个都生成了，再跑完整流程。

### 0.6 从中间结果继续跑

如果 Step1 已经跑完，要从 Step2-3 后继续全流程，可以直接在交互模式说：

```text
请从/Users/mkbk/PycharmProjects/new/program/output/step1_results继续处理，然后任务裁剪定为肝癌诊断
```

如果 Step2-3 已经跑完，要从 Step4 继续：

```text
请从/Users/mkbk/PycharmProjects/new/program/output/step2_3_results/next_input/input.csv继续处理，然后任务裁剪定为肝癌诊断
```

如果 Step4 已经跑完，要从 Step5 继续：

```text
请从/Users/mkbk/PycharmProjects/new/program/output/step4_results/next_input/filtered.csv继续处理，然后任务裁剪定为肝癌诊断
```

如果不确定该怎么说，最稳的是重新运行交互模式，然后直接说：

```text
请处理<你的原始数据目录>这个数据集，然后任务裁剪定为<你的任务目标>
```

## 1. 项目能做什么

主流程如下：

```text
原始数据目录
  ↓
Step1 数据重组
  - 扫描原始文件
  - 判断文件模态：table / figure / ocr
  - 拆分表格
  - 生成 records.json
  - 重组到 program/output/step1_results
  ↓
Step2-3 结构化与实体抽取
  - 扫描 Step1 输出目录
  - OCR / 文本抽取 / 表格读取
  - 聚合为 admission_wide CSV
  - 生成下一步 input.csv
  ↓
Step4 任务裁剪
  - 根据用户任务目标选择相关列
  - 输出 filtered.csv 和 selection_report.json
  ↓
Step5 数据清洗
  - 列画像
  - 风险分类
  - 生成/执行清洗脚本
  - 输出清洗后的 filtered.csv
  ↓
Step6 患者级一致性验证
  - 按患者聚合检查一致性
  - 生成 passed_patients_*.json
  ↓
Step7 ML 训练数据生成
  - 根据任务目标构造标签
  - 输出 ml_dataset_*.csv/json/jsonl 和 model_config_*.json
```

主入口：

```bash
main_orchestrator/main_orchestrator.py
```

## 2. 目录说明

关键目录如下：

```text
.
├── main_orchestrator/          # 主编排 Agent，负责路由、handoff、Step 顺序控制
├── configs/                    # 模型配置模板和配置加载器
├── step-1/                     # Step1 数据重组运行时和 Agent
├── step-2-3/                   # Step2-3 结构化运行时和 Agent
├── step-4/                     # Step4 任务裁剪运行时和 Agent
├── step-5/                     # Step5 数据清洗运行时和 Agent
├── step-6/                     # Step6 患者一致性验证运行时和 Agent
├── step-7/                     # Step7 ML 数据集生成运行时和 Agent
├── agent_1/                    # Step1 旧入口和 DocLayout 工具
├── agent_2-3/                  # OCR、实体抽取、结构化工具
├── agent_4/                    # 医疗列选择器核心实现
├── agent_5/                    # 数据质量修复核心实现
├── agent_6-7/                  # Step6/7 旧版兼容目录、ICD 资源
├── memory_agent/               # ReMeLight / Task / Tool memory 服务
├── program/output/             # 运行产物，默认被 .gitignore 忽略
├── reorganized_output/         # Step1 records.json 等中间产物，默认被 .gitignore 忽略
├── rawdata/                    # 本地原始数据示例目录，默认不提交
├── mimic-10/                   # 本地测试数据，默认不提交
└── mimic-mini/                 # 本地测试数据，默认不提交
```

注意：`rawdata/`、`mimic-10/`、`mimic-mini/`、`program/output/`、`reorganized_output/` 都是本地数据或运行产物，不应该提交到 Git。

## 3. 运行前必须准备

### 3.1 Python 版本

推荐使用 Python 3.10。

当前项目在本机稳定使用：

```bash
/opt/anaconda3/envs/py310/bin/python
```

建议新机器这样建环境：

```bash
conda create -n py310 python=3.10 -y
conda activate py310
python --version
```

不要直接用系统 Python 或 base Python 3.13。某些 AgentScope 版本在 Python 3.13 下可能缺少 `CharTokenCounter`，会在导入 Step6/Step7 时失败。

### 3.2 Python 依赖

这个仓库目前没有统一的根目录 `requirements.txt`，因此第一次部署建议安装核心依赖和可选依赖。

核心依赖：

```bash
pip install \
  agentscope \
  pandas \
  numpy \
  pyyaml \
  python-dotenv \
  openpyxl \
  tqdm \
  pydantic \
  pytest \
  pytest-asyncio
```

Step2-3 OCR 相关依赖：

```bash
pip install rapidocr_onnxruntime
```

Step1 DocLayout-YOLO 相关依赖：

```bash
pip install opencv-python pillow torch thop
```

如果你的环境能安装 `doclayout_yolo`，再安装：

```bash
pip install doclayout-yolo
```

Step7 ICD 语义检索可选依赖：

```bash
pip install sentence-transformers torch rapidfuzz scikit-learn
```

如果只想先跑主链路基础流程，可以先安装核心依赖和 `rapidocr_onnxruntime`。DocLayout 和 ICD BERT 都有降级路径，但效果会下降。

### 3.3 模型 API 配置

模型配置有两种方式：

1. 环境变量。
2. 本地 YAML：`configs/model_config.local.yaml`。

推荐用本地 YAML，因为不同 Agent 可以使用不同模型和 endpoint。

先复制模板：

```bash
cp configs/model_config.yaml configs/model_config.local.yaml
```

`configs/model_config.local.yaml` 已经被 `.gitignore` 忽略，可以放心写真实 key，不要提交。

最小配置示例：

```yaml
global:
  api_key: ""
  base_url: ""
  model: ""
  timeout: 120
  default_temperature: 0.0
  default_seed: 666

agents:
  main_orchestrator:
    api_key: "你的主编排模型 API Key"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4.1-mini"
    temperature: 0.0
    seed: 666

  agent_1:
    api_key: "你的 Step1 模型 API Key"
    base_url: "https://api.minimaxi.com/v1"
    model: "MiniMax-M2.7"
    temperature: 0.0
    seed: 666

  agent_2_3:
    api_key: "你的 Step2-3 模型 API Key"
    base_url: "https://api.minimaxi.com/v1"
    model: "MiniMax-M2.7"
    temperature: 0.0
    seed: 666
    embedding:
      api_key: "你的 embedding API Key"
      base_url: "https://api.openai.com/v1"
      model: "text-embedding-3-small"

  agent_4:
    api_key: "你的 Step4 模型 API Key"
    base_url: "https://api.minimaxi.com/v1"
    model: "MiniMax-M2.7"
    temperature: 0.2
    seed: 666

  agent_5:
    api_key: "你的 Step5 模型 API Key"
    base_url: "http://127.0.0.1:8317/v1"
    model: "gpt-5.1"
    temperature: 0.0
    seed: 666

  agent_6_7:
    api_key: "你的 Step6/7 模型 API Key"
    base_url: "https://api.minimaxi.com/v1"
    model: "MiniMax-M2.7"
    temperature: 0.0
    seed: 666

  memory_agent:
    api_key: "你的 Memory 模型 API Key"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4.1-mini"
    temperature: 0.0
    seed: 666
    embedding:
      api_key: "你的 memory embedding API Key"
      base_url: "https://api.openai.com/v1"
      model: "text-embedding-3-small"
      dimensions: 1536
```

如果只想快速跑通，也可以只设置通用环境变量：

```bash
export OPENAI_API_KEY="你的 API Key"
export OPENAI_API_BASE="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
```

但注意：Step2-3、Step4、Step5、Step6/7 可能更适合独立配置，因为它们可能使用不同供应商、不同模型和不同并发策略。

### 3.4 配置读取优先级

通用 Agent 模型创建逻辑大致按这个顺序取值：

```text
环境变量 OPENAI_API_KEY / OPENAI_API_BASE / MODEL_NAME
  ↓
configs/model_config.local.yaml
  ↓
configs/model_config.yaml
  ↓
代码里的默认值
```

Step2-3 还支持专用环境变量：

```bash
export AGENT_2_3_API_KEY="..."
export AGENT_2_3_BASE_URL="..."
export AGENT_2_3_MODEL="..."
export AGENT_2_3_EMBEDDING_API_KEY="..."
export AGENT_2_3_EMBEDDING_BASE_URL="..."
export AGENT_2_3_EMBEDDING_MODEL="text-embedding-3-small"
```

Step6/7 支持：

```bash
export DATA_SOURCE="original"       # original 或 mimic
export PATIENT_ID_COL="patient_id"  # 不设置时会自动探测
export STEP6_CONFIDENCE_THRESHOLD="0.70"
```

### 3.5 本地权重和资源文件

#### Step1 DocLayout-YOLO 权重

如果要让 Step1 对图片/PDF 做 DocLayout-YOLO 版式判断，需要准备权重：

```text
agent_1/doclayout_yolo/model/doclayout_yolo_docstructbench_imgsz1024.pt
agent_1/doclayout_yolo/model/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt
```

这些 `.pt` 文件很大，默认被 `.gitignore` 忽略，不随项目提交。

如果没有权重，或者只想先跑数据链路，可以关闭 DocLayout：

```bash
export STEP1_DOCLAYOUT_ENABLED=0
```

关闭后，Step1 会使用路径和后缀做 fallback 模态判断：

- 路径里包含 `images` 的视觉文件更可能被判成 `figure`
- 其他视觉文件默认走 `ocr`
- 表格后缀直接判成 `table`

#### Step7 ICD-10 资源

Step7 诊断任务会用 ICD-10 资源：

```text
step-7/ICD-10.xlsx
```

可选的 ICD BERT 向量库：

```text
step-7/models/bge-small-zh-v1.5/
step-7/data/icd10_bert_index/
```

如果没有 BERT 向量库，可以关闭语义检索：

```bash
export ICD10_USE_BERT=0
```

关闭后 Step7 仍会尝试精确匹配、规范化匹配、模糊匹配和 LLM 候选选择，但诊断编码能力会弱一些。

### 3.6 输入数据准备

主链路接受一个原始数据目录。目录里可以包含：

- `.csv`
- `.tsv`
- `.xls`
- `.xlsx`
- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.tiff`
- `.tif`
- `.pdf`

Step1 会从路径或文件名中尽量识别患者 ID。常见可识别形式包括：

```text
p10000032/
patient-10000032/
10000032/
```

MIMIC 风格数据通常会包含：

```text
images/
table/
notes/
structured/
```

项目不会自动下载 MIMIC 数据。你需要自己准备本地数据目录，例如：

```text
/Users/yourname/PycharmProjects/new/mimic-mini
/Users/yourname/PycharmProjects/new/mimic-10
```

## 4. 快速启动

### 4.1 进入项目和环境

```bash
cd /Users/mkbk/PycharmProjects/new
conda activate py310
```

如果不是这台机器，把路径换成你的项目路径。

### 4.2 运行交互模式

```bash
python main_orchestrator/main_orchestrator.py
```

进入交互后，输入类似：

```text
请处理/Users/mkbk/PycharmProjects/new/mimic-mini这个数据集，然后任务裁剪定为肝癌诊断
```

如果主链路判断任务需要 `task_text`，会提示：

```text
[MultiAgent] 请输入本次任务裁剪的目标:
```

这里填：

```text
肝癌诊断
```

### 4.3 一条命令运行 full pipeline

```bash
python main_orchestrator/main_orchestrator.py \
  "/Users/mkbk/PycharmProjects/new/mimic-mini" \
  --task-type full_pipeline
```

如果命令里没有明确任务目标，Step4/Step7 需要目标时可能会失败或在交互模式询问。推荐在交互输入中写清楚：

```text
请运行 full-pipeline，处理 /Users/mkbk/PycharmProjects/new/mimic-mini，本次任务裁剪的目标是肝癌诊断
```

### 4.4 禁用 MemoryAgent

MemoryAgent 默认启用。它会给 Orchestrator 提供历史经验检索和运行后摘要写入能力，但不改变 Step 顺序和路径。

如果想减少变量，先跑主链路，可以关闭：

```bash
python main_orchestrator/main_orchestrator.py \
  "/Users/mkbk/PycharmProjects/new/mimic-mini" \
  --task-type full_pipeline \
  --disable-memory-agent
```

### 4.5 输出最终 JSON

```bash
python main_orchestrator/main_orchestrator.py \
  "/Users/mkbk/PycharmProjects/new/mimic-mini" \
  --task-type full_pipeline \
  --disable-memory-agent \
  --json
```

`--json` 会关闭一部分 progress 输出，便于被脚本调用。

## 5. 主链路任务类型

`--task-type` 可选值来自 `main_orchestrator/core/contracts.py`：

```text
full_pipeline
step1_only
step2_3_only
step4_only
step5_only
step6_only
step7_only
resume_from_records
resume_from_step2_3
resume_from_step4
resume_from_step5
resume_from_step6
repair_task
memory_only
```

常用的是：

| task-type | 用途 |
|---|---|
| `full_pipeline` | 从原始目录开始完整跑 Step1 到 Step7 |
| `step1_only` | 只做 Step1 数据重组 |
| `resume_from_records` | 用已有 `records.json` 重组输出 |
| `resume_from_step2_3` | 从 Step4 开始继续跑到 Step7 |
| `resume_from_step4` | 从 Step5 开始继续跑到 Step7 |
| `resume_from_step5` | 从 Step6 开始继续跑到 Step7 |
| `resume_from_step6` | 只继续 Step7 |

## 6. 各 Step 输入输出

### 6.1 Step1 数据重组

输入：

```text
原始数据目录
```

输出：

```text
reorganized_output/_meta/records.json
step-1/generated_reorganizer.py
program/output/step1_results/
```

单独运行：

```bash
python agent_1/main.py \
  /Users/mkbk/PycharmProjects/new/mimic-mini \
  --output-root /Users/mkbk/PycharmProjects/new/program/output/step1_results \
  --records-path /Users/mkbk/PycharmProjects/new/reorganized_output/_meta/records.json \
  --script-path /Users/mkbk/PycharmProjects/new/step-1/generated_reorganizer.py \
  --json
```

重要环境变量：

```bash
export STEP1_MAX_WORKERS=4
export STEP1_REORGANIZE_MAX_WORKERS=4
export STEP1_DOCLAYOUT_ENABLED=1
export STEP1_DOCLAYOUT_BATCH_SIZE=16
export STEP1_PROGRESS_ENABLED=true
```

### 6.2 Step2-3 结构化

输入：

```text
program/output/step1_results/
```

输出：

```text
program/output/step2_3_results/results_<timestamp>/
program/output/step2_3_results/next_input/input.csv
program/output/step2_3_results/next_input/summary.json
```

主链路会自动把 Step1 的输出目录传给 Step2-3。

Step2-3 关键参数：

```text
ocr_workers: 默认 8
concurrency: 默认 8
mode: auto / directory / ocr_fill
```

常用环境变量：

```bash
export RAPIDOCR_USE_GPU=0
export RAPIDOCR_GPU_IDS="0"
export SKIP_FOLDERS="分割"
export STEP23_PROGRESS_ENABLED=true
```

如果 GPU 配置不稳定，建议先用 CPU：

```bash
export RAPIDOCR_USE_GPU=0
```

### 6.3 Step4 任务裁剪

输入：

```text
program/output/step2_3_results/next_input/input.csv
```

必须提供任务目标，例如：

```text
死亡预测
肝癌诊断
Dix-Hallpike 检查完成度预测
```

输出：

```text
program/output/step4_results/input_filtered_<timestamp>.csv
program/output/step4_results/input_selection_report_<timestamp>.json
program/output/step4_results/next_input/filtered.csv
program/output/step4_results/next_input/selection_report.json
```

单独运行：

```bash
python agent_4/main.py \
  --input /Users/mkbk/PycharmProjects/new/program/output/step2_3_results/next_input/input.csv \
  --output /Users/mkbk/PycharmProjects/new/program/output/step4_results \
  --task-text "肝癌诊断"
```

注意：诊断类任务如果后续 Step7 要构造 ICD 标签，Step4 必须保留能表达诊断的列，例如 ICD code、诊断文本、实体诊断字段。否则 Step7 会因为缺少标签来源返回 `NEEDS_REPAIR`。

### 6.4 Step5 数据清洗

输入：

```text
program/output/step4_results/next_input/filtered.csv
```

输出：

```text
program/output/step5_results/cleaned_<timestamp>.csv
program/output/step5_results/column_profile_<timestamp>.json
program/output/step5_results/column_risk_report_<timestamp>.json
program/output/step5_results/generated_scripts/
program/output/step5_results/next_input/filtered.csv
```

单独运行：

```bash
python agent_5/main.py \
  --input /Users/mkbk/PycharmProjects/new/program/output/step4_results/next_input/filtered.csv \
  --output /Users/mkbk/PycharmProjects/new/program/output/step5_results \
  --workers 4 \
  --llm-workers 2
```

禁用 LLM 清洗脚本生成：

```bash
python agent_5/main.py \
  --input /Users/mkbk/PycharmProjects/new/program/output/step4_results/next_input/filtered.csv \
  --output /Users/mkbk/PycharmProjects/new/program/output/step5_results \
  --disable-llm
```

常用环境变量：

```bash
export STEP5_PROGRESS_ENABLED=true
```

### 6.5 Step6 患者一致性验证

输入：

```text
program/output/step5_results/next_input/filtered.csv
```

输出：

```text
program/output/step6_results/consistency_report_<timestamp>.txt
program/output/step6_results/consistency_report_<timestamp>.json
program/output/step6_results/passed_patients_<timestamp>.json
```

单独运行：

```bash
python step-6/run_step6_ml_pipeline.py \
  --input /Users/mkbk/PycharmProjects/new/program/output/step5_results/next_input/filtered.csv \
  --output /Users/mkbk/PycharmProjects/new/program/output/step6_results \
  --patient-id-col patient_id \
  --data-source original
```

复用已有报告：

```bash
python step-6/run_step6_ml_pipeline.py \
  --input /Users/mkbk/PycharmProjects/new/program/output/step5_results/next_input/filtered.csv \
  --output /Users/mkbk/PycharmProjects/new/program/output/step6_results \
  --reuse-existing
```

常用环境变量：

```bash
export STEP6_CONFIDENCE_THRESHOLD=0.70
export DATA_SOURCE=original
export PATIENT_ID_COL=patient_id
```

### 6.6 Step7 ML 数据集生成

输入必须同时具备：

```text
Step5 filtered.csv
Step4 selection_report.json
Step6 passed_patients_*.json
task_text
```

默认主链路会传入：

```text
program/output/step5_results/next_input/filtered.csv
program/output/step4_results/next_input/selection_report.json
program/output/step6_results/passed_patients_<timestamp>.json
本轮任务目标 task_text
```

输出：

```text
program/output/step7_results/ml_dataset_<task>_<timestamp>.csv
program/output/step7_results/ml_dataset_<task>_<timestamp>.json
program/output/step7_results/ml_dataset_<task>_<timestamp>.jsonl
program/output/step7_results/model_config_<task>_<timestamp>.json
```

单独运行：

```bash
python step-7/run_step7_ml_pipeline.py \
  --input /Users/mkbk/PycharmProjects/new/program/output/step5_results/next_input/filtered.csv \
  --selection-report /Users/mkbk/PycharmProjects/new/program/output/step4_results/next_input/selection_report.json \
  --passed-patients-json /Users/mkbk/PycharmProjects/new/program/output/step6_results/passed_patients_20260523_154001.json \
  --task-text "肝癌诊断" \
  --output /Users/mkbk/PycharmProjects/new/program/output/step7_results
```

常用环境变量：

```bash
export ICD10_USE_BERT=1
export ICD10_BERT_MODEL="BAAI/bge-small-zh-v1.5"
export ICD10_BERT_MIN_SIM=0.70
export DATA_SOURCE=original
export PATIENT_ID_COL=patient_id
```

Step7 如果返回 `NEEDS_REPAIR`，最常见原因不是程序崩了，而是当前数据不支持任务标签构造。例如任务是“肝癌诊断”，但 Step4 筛选后的表里没有 ICD code、HCC、liver cancer、C22 相关列，那么 Step7 无法生成监督学习标签。

## 7. MemoryAgent 配置

MemoryAgent 默认启用。它的职责是：

- 给 Orchestrator 提供历史规则和 Step 经验。
- 记录本次运行的紧凑摘要。
- 不决定主 pipeline 的 Step 顺序。
- 不修改 TaskSpec、路径、allowed_steps 或 required_steps。

配置位置：

```text
configs/model_config.local.yaml -> agents.memory_agent
```

最小配置：

```yaml
agents:
  memory_agent:
    api_key: "你的 Memory 模型 API Key"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4.1-mini"
    temperature: 0.0
    seed: 666
    embedding:
      api_key: "你的 embedding API Key"
      base_url: "https://api.openai.com/v1"
      model: "text-embedding-3-small"
      dimensions: 1536
```

查看 MemoryAgent 状态：

```bash
python memory_agent/main_memory.py --status
```

关闭 MemoryAgent：

```bash
python main_orchestrator/main_orchestrator.py --disable-memory-agent
```

MemoryAgent 运行产物默认在：

```text
memory_agent/reme_memory/playbook_light/
memory_agent/reme_memory/vector/
```

这些目录被忽略，不应提交。

## 8. 推荐运行顺序

第一次部署建议不要直接跑大数据。先按下面顺序验证。

### 8.1 检查 Python 和依赖

```bash
which python
python --version
python -m pip --version
python -m py_compile \
  main_orchestrator/main_orchestrator.py \
  main_orchestrator/agents/orchestrator_agent.py \
  step-6/step6_runtime.py \
  step-7/step7_runtime.py
```

### 8.2 跑测试

```bash
python -m pytest main_orchestrator/tests -q
python -m pytest memory_agent/tests -q
```

### 8.3 用小数据跑 Step1

```bash
export STEP1_DOCLAYOUT_ENABLED=0

python main_orchestrator/main_orchestrator.py \
  "/Users/mkbk/PycharmProjects/new/mimic-mini" \
  --task-type step1_only \
  --disable-memory-agent \
  --json
```

检查：

```text
reorganized_output/_meta/records.json
program/output/step1_results/
```

### 8.4 再跑完整链路

```bash
python main_orchestrator/main_orchestrator.py
```

交互输入：

```text
请处理/Users/mkbk/PycharmProjects/new/mimic-mini这个数据集，然后任务裁剪定为肝癌诊断
```

## 9. 输出目录总览

默认输出位置：

```text
reorganized_output/_meta/records.json
step-1/generated_reorganizer.py
program/output/step1_results/
program/output/step2_3_results/
program/output/step4_results/
program/output/step5_results/
program/output/step6_results/
program/output/step7_results/
```

每一步的标准下游输入一般在：

```text
program/output/<step>_results/next_input/
```

其中：

```text
Step2-3 -> next_input/input.csv
Step4   -> next_input/filtered.csv + selection_report.json
Step5   -> next_input/filtered.csv
Step6   -> passed_patients_*.json
Step7   -> ml_dataset_* + model_config_*
```

## 10. 常见问题

### 10.1 报错：缺少 API key

典型提示：

```text
请设置 OPENAI_API_KEY 或 configs/model_config.yaml 中 agents.xxx.api_key
```

处理：

1. 确认是否创建了 `configs/model_config.local.yaml`。
2. 确认对应 agent 的 `api_key` 不为空。
3. 如果用环境变量，确认当前 shell 里能看到：

```bash
echo $OPENAI_API_KEY
echo $OPENAI_API_BASE
echo $MODEL_NAME
```

### 10.2 报错：invalid params, user name must be consistent

这是部分 OpenAI-compatible 后端对 `user.name` 字段严格导致的。当前主链路已经在 `main_orchestrator/agents/agent_runtime.py` 中把协议级 name 固定为 `user`，如果仍然出现，优先确认你运行的是当前仓库代码，而不是旧副本。

### 10.3 Step2-3 还没完，为什么 Step6/7 出来了

正常主链路会严格按：

```text
step1 -> step2_3 -> step4 -> step5 -> step6 -> step7
```

如果日志里 Step6/7 提前出现，先确认是不是旧进程输出、多个终端同时运行、或者上一轮残留日志。当前 Orchestrator 会在每次 handoff 前打印：

```text
[Orchestrator] DECISION handoff -> StepX
[Orchestrator] START StepX
[Orchestrator] END StepX
```

以这个为准判断真实顺序。

### 10.4 Step7 返回 NEEDS_REPAIR

Step7 的 `NEEDS_REPAIR` 通常表示数据不能支持目标任务，不一定是代码崩溃。

例子：

```text
任务目标：肝癌诊断
但筛选后数据没有 ICD code / HCC / liver cancer / C22 相关字段
```

这种情况下 Step7 无法构造监督标签。处理方式：

1. 回到 Step4，检查 `selection_report.json`。
2. 确认是否保留 `diagnoses_icd`、`icd_code`、诊断文本、实体诊断列。
3. 必要时强化 Step4 的任务裁剪策略，让诊断任务强保留标签候选列。

### 10.5 Step6 保存通过患者数量不对

检查：

```text
program/output/step6_results/passed_patients_<timestamp>.json
```

如果报告里说通过 10 位，但 JSON 只有 1 位，说明报告标记或缓存写入异常。应重新跑 Step6，避免 Step7 读取错误的 passed patient 列表。

### 10.6 终端打印 `<think>...</think>`

当前 Step6/Step7 runtime 已经对最终展示文本做 `<think>` 清洗。如果仍然看到：

1. 确认运行的是当前分支当前文件。
2. 确认不是 AgentScope 内部 debug 输出。
3. 确认没有从旧日志复制输出。

### 10.7 大量 `DtypeWarning`

这是 pandas 读取宽表 CSV 时列类型混合导致，通常不阻塞流程。

如果需要减少警告，可以在相关读取逻辑中使用 `low_memory=False`，但当前 warning 本身不代表任务失败。

### 10.8 OCR 慢或连接重试

Step2-3 中可能看到：

```text
[Retry] 连接错误，10s 后重试
```

这通常是 LLM endpoint 或 OCR/抽取阶段并发过高导致。处理：

1. 降低并发。
2. 检查 API endpoint 是否稳定。
3. 先用小数据跑通。

### 10.9 DocLayout-YOLO 权重缺失

如果没有权重，先关闭：

```bash
export STEP1_DOCLAYOUT_ENABLED=0
```

等主链路跑通后再补权重和相关依赖。

### 10.10 ICD BERT 加载失败

如果看到：

```text
使用 ICD BERT 检索需要安装: pip install sentence-transformers torch
```

可以安装依赖，或先关闭：

```bash
export ICD10_USE_BERT=0
```

## 11. 安全和提交规则

不要提交：

```text
.env
configs/model_config.local.yaml
rawdata/
rawdata-1/
mimic-10/
mimic-mini/
program/output/
reorganized_output/
agent_1/doclayout_yolo/model/*.pt
step-7/models/
step-7/data/icd10_bert_index/
memory_agent/reme_memory/
*.csv
*.xlsx
*.npy
```

提交前检查：

```bash
git status --short
git diff --cached --name-only
git grep -n "password:\\|TOKEN\\|SECRET\\|Bearer \\|api_key" -- .
```

如果 `git grep` 命中的是模板里的空字段，需要人工确认；如果命中真实 key，必须移除后再提交。

## 12. 给新使用者的最小 checklist

把项目交给别人时，至少告诉对方：

```text
1. 使用 Python 3.10，不要用系统 Python 3.13。
2. 先安装核心依赖和 OCR 依赖。
3. 复制 configs/model_config.yaml 为 configs/model_config.local.yaml。
4. 在 model_config.local.yaml 填各 Agent 的 api_key、base_url、model。
5. 准备本地输入数据目录，例如 mimic-mini。
6. 如果没有 DocLayout 权重，先 export STEP1_DOCLAYOUT_ENABLED=0。
7. 如果没有 ICD BERT 向量库，先 export ICD10_USE_BERT=0。
8. 先跑 main_orchestrator/tests 和 memory_agent/tests。
9. 先用小数据跑 step1_only。
10. 再跑 full_pipeline。
```

最小 smoke test：

```bash
cd /Users/mkbk/PycharmProjects/new
conda activate py310
export STEP1_DOCLAYOUT_ENABLED=0
export ICD10_USE_BERT=0

python -m pytest main_orchestrator/tests -q
python -m pytest memory_agent/tests -q

python main_orchestrator/main_orchestrator.py \
  "/Users/mkbk/PycharmProjects/new/mimic-mini" \
  --task-type step1_only \
  --disable-memory-agent \
  --json
```

完整交互运行：

```bash
python main_orchestrator/main_orchestrator.py
```

然后输入：

```text
请处理/Users/mkbk/PycharmProjects/new/mimic-mini这个数据集，然后任务裁剪定为肝癌诊断
```

## 13. 当前已知限制

1. 根目录暂时没有统一 `requirements.txt`，新环境需要按本文档安装依赖。
2. Step7 对任务标签来源很敏感，Step4 如果把 ICD/诊断列裁掉，Step7 会无法构造标签。
3. DocLayout-YOLO 权重和 ICD BERT 向量库是本地大文件，不随仓库提交。
4. MemoryAgent 是辅助上下文系统，不保证每次检索都有有效历史经验。
5. 医疗数据目录结构差异很大，新数据集第一次运行建议先跑 Step1 和 Step2-3，不要直接跑大规模 full pipeline。
