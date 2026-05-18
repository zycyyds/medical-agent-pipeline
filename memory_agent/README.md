# Memory Agent

`memory_agent` 现在按真实 ReMe 优先的双层记忆系统运行：

- `ReMeLight`：规则记忆，替代旧 Playbook，保存流程约束、失败教训、验证口径和工具使用原则。
- `ReMe Task/Tool Memory`：经验记忆，保存 Step1/Step4 相似任务案例和细粒度工具调用经验。

主链路仍然只通过 `main_orchestrator/memory/memory_supervisor.py` 调用两个稳定接口：

```python
context_str, used_memory_ids = await get_pipeline_context(query_text="处理 mimic-10，只跑 step1")
summary = await report_pipeline_result(trace_payload, used_memory_ids=used_memory_ids)
```

MemoryAgent 只做检索、总结和写入记忆，不决定主 pipeline 的 Step 调度。

## 运行

```bash
python memory_agent/main_memory.py
python memory_agent/main_memory.py --status
```

主链路默认启用 MemoryAgent：

```bash
python main_orchestrator/main_orchestrator.py
```

关闭 MemoryAgent：

```bash
python main_orchestrator/main_orchestrator.py --disable-memory-agent
```

## 配置

配置从 `configs/model_config.local.yaml` 的 `memory_agent` 读取：

```yaml
memory_agent:
  api_key: "..."
  base_url: "https://api.minimaxi.com/v1"
  model: "MiniMax-M2.7"
  temperature: 0.0
  seed: 666

  reme_enabled: true
  reme_backend: "real"
  reme_light_root: "memory_agent/reme_memory/playbook_light"
  reme_vector_root: "memory_agent/reme_memory/vector"

  embedding:
    api_key: "..."
    base_url: "https://yunwu.ai/v1"
    model: "text-embedding-3-small"
    dimensions: 1536
```

如果 `memory_agent.embedding` 没配置，当前实现会兼容读取 `agent_2_3.embedding`。

## 存储边界

- ReMeLight 规则记忆默认写入 `memory_agent/reme_memory/playbook_light/`。
- ReMe Task/Tool 经验记忆默认写入 `memory_agent/reme_memory/vector/`。
- 运行时生成的 ReMe 目录已在 `memory_agent/reme_memory/.gitignore` 中忽略。

## Step 经验记录

会记录：

- Step1 input profile：文件数、后缀分布、schema 摘要、路径类型。
- Step1 结果：records 数、模态分布、表格拆分数、输出文件数、validator 状态。
- Step1 工具经验：关键工具调用是否成功、输出摘要和参数画像。
- Step1 generated script 的路径、hash 和大小。
- Step4 输入表画像：文件大小、列数、样例列、profile hash。
- Step4 任务裁剪经验：任务文本、词表 profile、结构化字段保护、最终列数和输出报告路径。

不会记录：

- 完整 `records.json`
- 完整 CSV 内容
- 图片正文
- 完整生成脚本正文
- API key
- 超长 terminal trace

## Legacy

旧 Playbook / Reflector / Curator 主路径、旧 JSONL cases fallback、旧
`reme_memory_agent.py` 兼容入口均已移除。当前唯一推荐入口是
`memory_agent/main_memory.py`，主链路只通过 `MemoryService` 访问 ReMeLight
和 ReMe Task/Tool Memory。
