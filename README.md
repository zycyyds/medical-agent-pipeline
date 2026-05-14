# Medical Agent Pipeline

多阶段医疗数据处理实验项目，包含 Step1 数据重组、Step2/3 结构化与初步清洗、Step4 任务裁剪、Step5 数据质量修复、Step6/7 一致性验证与建模数据集生成，以及 Playbook/ReMe 记忆模块。

## Repository Scope

这个公开仓库只保存代码和安全的配置模板，不保存本地数据、运行产物、模型权重或真实密钥。

已排除的内容包括：

- `rawdata/`、`rawdata-1/`、`mimic-10/`、`mimic-mini/`
- `program/output/`、`reorganized_output/`、`output/`
- `agent_1/doclayout_yolo/model/*.pt`
- `agent_6-7/models/`
- `.env`、`configs/model_config.local.yaml`

## Configuration

公开的 `configs/model_config.yaml` 是模板，不包含真实 API key。真实配置建议放在环境变量或本地忽略文件中：

```bash
export OPENAI_API_KEY="..."
export OPENAI_API_BASE="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
```

如果需要保留本地 YAML 配置，可以复制为：

```bash
cp configs/model_config.yaml configs/model_config.local.yaml
```

`configs/model_config.local.yaml` 已被 `.gitignore` 忽略。

## Step1 Smoke Test

Step1 支持从原始目录生成 `records.json`，并可以从 `records.json` 续跑。示例：

```bash
python main_orchestrator/main_orchestrator.py mimic-10 --task-type step1_only --disable-memory-agent --json
python main_orchestrator/main_orchestrator.py reorganized_output/_meta/records.json --task-type resume_from_records --disable-memory-agent --json
```

如果启用 DocLayout-YOLO，需要把权重文件放到：

```text
agent_1/doclayout_yolo/model/
```

权重文件不随仓库提交。

## Tests

核心回归测试：

```bash
python -m pytest agent_1/tests main_orchestrator/tests -q
```

上传前建议做敏感信息检查，确保没有真实密钥或数据文件进入 Git：

```bash
git diff --cached --name-only
git grep -n "password:\\|TOKEN\\|SECRET\\|Bearer " -- .
```
