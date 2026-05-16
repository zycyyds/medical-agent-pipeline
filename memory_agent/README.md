# Memory Agent

`memory_agent` 现在分成两条记忆线：

- `playbook/`：策略本记忆，记录可复用规则、失败教训和反思结果。
- `reme/`：案例型记忆，记录类似 Step1 脚本案例的可检索元数据。

当前主链路只接入 Playbook。ReMe 已作为独立能力保留，暂不自动影响 main pipeline。

它承担三件事：

1. 在任务开始前，按当前输入检索最相关的经验条目并注入给 Generator。
2. 在任务结束后，消费结构化 trace，判断哪些 bullet 有帮助、哪些有害。
3. 把新经验以 curator delta 的形式写回 Playbook，并执行 grow-and-refine。

## 当前主接口

```python
from memory_agent.memory_tool import get_playbook_context_data

context_str, used_ids = get_playbook_context_data(
    query_text="../rawdata/垂直眼位",
    max_bullets=15,
)
```

ReMe 独立接口：

```python
from memory_agent.reme.tools import (
    initialize_reme_memory_store,
    build_step1_input_profile,
    record_step1_script_case,
    retrieve_step1_script_case,
)

initialize_reme_memory_store()
profile = build_step1_input_profile("mimic-10")
```

```python
import asyncio
from memory_agent.memory_tool import curator_process_execution

result = asyncio.run(curator_process_execution(trace_json))
print(result.content)
```

## 目录

```text
memory_agent/
├── __init__.py
├── ace_tools.py              # 兼容层，旧导入路径仍可用
├── config.py
├── main_memory.py
├── memory_tool.py            # 主工具入口
├── memory_bank.json
├── agentscope_tools.json
├── playbook/                 # Playbook 命名空间
├── reme/                     # ReMe 命名空间
├── reme_memory/              # ReMe JSONL fallback 存储目录
├── reme_memory_agent.py      # ReMe 独立交互入口
├── reme_memory_tool.py       # ReMe 工具实现
└── core/
    ├── __init__.py
    ├── curator.py
    ├── models.py
    ├── playbook.py
    ├── reflector.py
    └── trace_parser.py
```

## 设计重点

- 记忆按 section 化 bullet 管理，而不是单段摘要。
- Reflector 只输出结构化 reflection，不直接改写 Playbook。
- Curator 只提增量 `ADD` 操作，真正写库由系统确定性完成。
- Playbook 检索按 query 排序，grow-and-refine 负责归并与归档低效条目。
- 兼容旧的自然语言 trace，但主链优先使用结构化 trace payload。

## ReMe 边界

- ReMe 保存案例元数据，不保存完整脚本正文。
- 默认使用 `memory_agent/reme_memory/cases/step1_script_cases.jsonl` 作为 JSONL fallback。
- 写入前会根据 input profile hash、script path、script hash 生成稳定 `case_id`，同一案例重复记录时返回 `EXISTS`，不会重复追加。
- 真实 `reme` / `reme_ai` 依赖可用时只在状态里报告，当前业务路径仍以 JSONL fallback 为稳定默认。
