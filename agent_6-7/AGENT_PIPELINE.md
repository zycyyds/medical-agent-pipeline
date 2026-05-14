# Step 6 & Step 7 Agent 流水线说明

本文档对应 `run_step6_7_ml_pipeline.py`：基于 **AgentScope** 的 `ReActAgent`，将医疗患者数据先做 **一致性验证（Step 6）**，再生成 **ML 训练数据（Step 7）**。

---

## 1. 整体架构

| 层级 | 职责 |
|------|------|
| **工具（Tools）** | 只负责读 CSV、写文件、调用确定性算法（如 ICD 匹配、统计）。**不内置**「哪些列算诊断、标签怎么定义」等业务结论。 |
| **LLM Agent** | 通过 ReAct 循环**自主调用工具**，根据工具返回结果推理：是否通过患者、选哪一列做标签、`label_mapping` 等。 |
| **Pipeline** | `run_pipeline()` 先跑 Step 6 Agent，从最终回复中解析 `PASSED_PATIENTS`，再跑 Step 7 Agent。 |

**设计原则**：领域判断（字段意义、标签策略）由 LLM 完成；数据访问与编码逻辑由 Python 函数保证可重复、可审计。

---

## 2. 技术栈

- **框架**：AgentScope — `ReActAgent`、`Toolkit`、`OpenAIChatModel`（兼容 OpenAI API 的网关，如 `OPENAI_BASE_URL`）。
- **数据**：`pandas` 读 CSV；`numpy` 用于 ICD 向量检索。
- **ICD 辅助**：`rapidfuzz`（字符串模糊）；可选 `sentence-transformers` + 预建向量库（BERT 语义检索）。
- **异步入口**：`asyncio.run(run_pipeline())`。

---

## 3. 数据路径与全局配置

| 配置项 | 含义 |
|--------|------|
| `DATA_DIR` | `agent_6-7/data_input/` |
| `RAW_CSV` | 默认读取 `data_input/raw.csv`；不存在时复用筛选后 CSV |
| `FILTERED_CSV` | 优先读取 `data_input/filtered.csv`，其次 `data_input/input.csv`，再取目录内最新 CSV |
| `SELECTION_REPORT` | 优先读取 `data_input/selection_report.json`，其次匹配最新 `*_selection_report.json` |
| `ICD10_XLSX` | 仓库根目录下 `ICD-10.xlsx`（国标诊断名称 ↔ 编码） |
| `ICD10_VECTOR_DIR` | `data/icd10_bert_index/`（`build_icd10_vector_store.py` 生成） |
| `CONFIDENCE_THRESHOLD` | `0.70`，Step 6 中「通过」的置信度下限 |
| `_passed_patient_ids` | Step 6 结束后由解析器写入，Step 7 过滤患者用 |

### 长文本列黑名单 `_SKIP_COLS`

以下列名在「给 LLM 看样本 / 自动特征列」时会被跳过，避免把整段 OCR、抽取 JSON 塞进提示词：

- `ocr_text`、`preprocessed_text`、`file_path`、`error`
- `*_抽取` / `*_标准化`（Test/Disease/Drug/Symptom 等）
- `relations`、`temporal_info`、`quantity_info`

---

## 4. Step 6：ConsistencyAgent（一致性验证）

### 4.1 角色与粒度

- **名称**：`ConsistencyAgent`
- **系统角色**：医疗数据质量工程师，做前庭相关数据集的一致性检查。
- **粒度**：**患者（patient_id）**。同一患者在 `RAW_CSV` 中可能有多行，工具按患者聚合分析。

### 4.2 工作流程（系统提示中约定）

1. `load_data_overview` — 总行数、患者数、列数、示例字段。
2. `get_patient_list` — 所有 `patient_id` 及每人记录数。
3. 对**每一位**患者调用 `analyze_patient_consistency(patient_id)`。
4. 按规则打 **0.0–1.0 置信度**，并与 `CONFIDENCE_THRESHOLD` 比较是否通过。
5. `save_step6_report` 保存报告；**正文末尾必须**包含一行供机器解析：

```text
PASSED_PATIENTS: [36906, 36907, ...]
```

### 4.3 Step 6 工具一览

| 工具 | 作用 |
|------|------|
| `load_data_overview` | 读 `RAW_CSV`，概况 + 第一条非跳过列的预览。 |
| `get_patient_list` | 每位患者记录数列表。 |
| `analyze_patient_consistency` | **核心**：该患者多行之间的跨字段不一致、数值波动、完整性。 |
| `save_step6_report` | 写 `output_step6/consistency_report_<时间戳>.txt` 与 `.json`。 |

### 4.4 `analyze_patient_consistency` 细节

对单个 `patient_id` 在 `RAW_CSV` 中筛出所有行后：

1. **跨记录不一致字段**  
   某列在非空情况下若出现 **≥2 种不同取值**，则记为不一致。  
   **严重程度**（仅用于摘要与 Agent 扣分）：  
   - 列名含 `诊断`、`性别`、`民族` → **HIGH**  
   - 列名含 `年龄` → **MEDIUM**  
   - 其余 → **LOW**

2. **数值字段波动**  
   能转为数值的列：若跨记录 `max - min > 0`，统计波动及相对波动率（用于报告）。

3. **字段完整性**  
   在排除 id/文件名/黑名单列后的「数据列」中，**至少有一条非空**的列数 ÷ 总列数 → 百分比。

Agent 侧扣分规则（系统提示）：HIGH ×0.20、MEDIUM ×0.10、LOW ×0.03、完整性 &lt;30% 额外 −0.10（具体以代码内 `sys_prompt` 为准）。

### 4.5 Step 6 输出

- **TXT**：人类可读报告 + `PASSED_PATIENTS`。  
- **JSON**：时间戳、`report_text`、摘要字段等。

**Step 7** 若 `_passed_patient_ids` 非空，则 **仅保留** 这些患者在 `FILTERED_CSV` 中的行；解析失败时退化为「全部患者」。

---

## 5. Step 7：DataPrepAgent（ML 数据准备）

### 5.1 角色

- **名称**：`DataPrepAgent`
- **系统角色**：ML 数据工程师，在**已通过 Step 6** 的子集上，结合 `task_text` 与列报告，产出带 ICD 增强的训练表。

### 5.2 工作流程（系统提示中约定）

1. `load_task_context` — `task_text`、`final_columns`、诊断相关列分布等。  
2. `discover_diagnosis_fields` — LLM 扫字段名+样本，猜哪些列像诊断。  
3. `extract_and_map_all_diagnosis_fields(diagnosis_fields=[...])` — 逐字段唯一值：先 LLM 抽疾病名，再 ICD 映射，得到 `icd10_mapping`。  
4. 选定 `label_column`，**必须**调用 `propose_label_mapping` 生成 `label_mapping`（**禁止**向终端用户追问映射）。  
5. `build_label_series` — 按**行**验证标签（与保存逻辑一致）。  
6. `format_and_save_ml_dataset` — 写 CSV / JSONL / JSON。  
7. 文字总结标签含义、ICD 覆盖、局限。

`max_iters=35`，减少步数用尽导致中断。

### 5.3 Step 7 工具一览

| 工具 | 作用 |
|------|------|
| `load_task_context` | 读 `SELECTION_REPORT` + `FILTERED_CSV` 摘要。 |
| `get_column_distribution` | 单列分布、缺失率、值频数等。 |
| `discover_diagnosis_fields` | LLM 识别诊断相关列名 + 列唯一值列表。 |
| `extract_and_map_all_diagnosis_fields` | 多字段批量：LLM 提取疾病 + `_local_icd10_lookup`。 |
| `map_diagnoses_to_icd10` | 单列诊断名列表 → ICD（已知是疾病名时用）。 |
| `propose_label_mapping` | **从数据自动生成** `label_mapping`（`multiclass_enum` 或 `binary_positive_vs_rest`）。 |
| `build_label_series` | 不保存，按**每条筛选记录一行**预览 `y`。 |
| `format_and_save_ml_dataset` | 生成最终数据集文件。 |

---

## 6. ICD-10 映射链路（核心实现细节）

入口函数：`_local_icd10_lookup`（对外别名 `_umls_lookup_icd10`）。**不调用** NLM/UMLS 在线 API，以本地 `ICD-10.xlsx` 为主。

### 6.1 顺序概览

1. **精确匹配**：诊断名 → 字典。  
2. **规范化匹配**：`_normalize_diag`（NFKC、去空格、去尾部英文括号缩写等）后再查。  
3. **LLM 中文标准化** `_llm_cn_normalize`：英文缩写（如 BPPV、UPVD）→ 标准中文短语，再重复 1–2。  
4. **BERT 向量检索**（若 `data/icd10_bert_index/` 存在且 `ICD10_USE_BERT` 开启）：  
   - 句向量与矩阵点积（L2 归一化后等价余弦相似度）。  
   - Top1 ≥ `ICD10_BERT_MIN_SIM`（默认 0.70）则采纳，来源标记 `BERT`。  
5. **模糊匹配** `rapidfuzz`：先用 `WRatio` 取候选，再用 **`fuzz.ratio` 重排**，减轻短子串误匹配（如「损伤」误配长句）。  
6. **ratio ≥ 85** 可自动采纳为 `FUZZY`。  
7. 否则 **LLM 候选选择**：在 BERT + 模糊合并去重后的候选里选序号或 0。

缓存：`_ICD10_CACHE`、`_LLM_DISEASE_EXTRACT_CACHE`、`_LLM_CN_NORMALIZE_CACHE` 等，避免重复调用 API。

### 6.2 向量库构建

```bash
python build_icd10_vector_store.py
```

可选从魔搭下载权重：

```bash
python download_bert_modelscope.py
export ICD10_BERT_MODEL=<本地模型目录或 HuggingFace 模型名>
```

**注意**：`meta.json` 中的模型应与运行时 `ICD10_BERT_MODEL` 一致，否则应重建向量库。

---

## 7. `format_and_save_ml_dataset` 细节

### 7.1 样本粒度（重要）

- **每一行筛选后记录 = 一条 ML 样本**（**不再**按患者对数值取均值、对类别取众数合并为一行）。  
- 列 **`sample_row_id`**：当前 `FILTERED_CSV` 在应用 Step6 患者过滤并重置索引后的 **0…N-1**，便于回溯。

### 7.2 标签行

- 每行读取 `label_column` 的原始值，用 `label_mapping` 得到 `y`。  
- 无法映射的行（`y` 为 NaN）在 `dropna(subset=["y"])` 时 **丢弃**。  
- 至少要有 **2 个不同类别** 的 `y`，否则工具返回错误、不保存。

### 7.3 诊断列增强

对启发式或 LLM 缓存认定的「诊断相关」特征列，按**该行**单元格调用 `_llm_extract_disease_content`（可缓存）+ ICD 查询，追加：

- `{列名片段}_疾病名称`  
- `{列名片段}_ICD10`  

并增加 **`label_ICD10`**（由标签原始字符串映射 ICD）。

### 7.4 输出文件

| 文件 | 内容 |
|------|------|
| `ml_dataset_<ts>.csv` | 宽表，含特征、`y`、`label_name`、`sample_row_id`、ICD 增强列等。 |
| `ml_dataset_<ts>.jsonl` | 每行一个 JSON：`sample_row_id`、`patient_id`、`features`、`label`、`label_name`、`label_ICD10`。 |
| `ml_dataset_<ts>.json` | 含 `granularity: "record"`、`feature_names`、`X`、`y`、`sample_row_ids`、`patient_ids`、`icd10_mapping` 等，便于简单加载。 |

---

## 8. 环境变量（建议）

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | 调用 LLM 的密钥（**务必用环境变量，勿提交仓库**）。 |
| `OPENAI_BASE_URL` | API 网关，默认兼容 OpenAI 格式。 |
| `MODEL_NAME` | 聊天模型名，如 `gpt-4.1-mini`。 |
| `ICD10_USE_BERT` | `0` / `false` 关闭 BERT 检索。 |
| `ICD10_BERT_MODEL` | Sentence-Transformers 模型 ID 或**本地目录**。 |
| `ICD10_BERT_MIN_SIM` | BERT Top1 自动采纳阈值，默认 `0.70`。 |

---

## 9. 运行方式

```bash
cd agent_6-7
python run_step6_7_ml_pipeline.py
```

依赖包括但不限于：`agentscope`、`pandas`、`openpyxl`（读 xlsx）、`openai`、`rapidfuzz`；BERT 路径需 `sentence-transformers`、`torch`。

---

## 10. 相关文件索引

| 文件 | 作用 |
|------|------|
| `run_step6_7_ml_pipeline.py` | 主程序：双 Agent + 全部工具与 ICD 逻辑。 |
| `build_icd10_vector_store.py` | 从 `ICD-10.xlsx` 离线建向量库。 |
| `download_bert_modelscope.py` | 从魔搭下载句向量模型到本地。 |
| `requirements_icd_bert.txt` | BERT 建库/推理依赖。 |
| `requirements_modelscope.txt` | 魔搭下载脚本依赖。 |
| `../program/output/step6_results/` | 一致性报告。 |
| `../program/output/step7_results/` | ML 数据集输出。 |

---

## 11. 已知局限与注意点

- Step 6 置信度由 **LLM 根据规则估算**，非严格统计检验。  
- ICD 标准库无法覆盖的**新造临床用语**可能为 `NOT_FOUND`。  
- 标签列若大量空值或未出现在 `label_mapping` 中，有效样本会明显减少。  
- `propose_label_mapping` 的 `multiclass_enum` 按**字典序**编码，**0/1 的语义**需结合 `label_mapping` 自行解读；二分类若需「某病=1」，应用 `binary_positive_vs_rest` 并准确列出 `positive_values` 字符串。

---

*文档版本：与 `run_step6_7_ml_pipeline.py` 当前实现同步整理，如有改动请以源码为准。*
