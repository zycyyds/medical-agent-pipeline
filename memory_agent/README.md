# Memory Agent

`memory_agent` 现在按 ACE 思路维护一个可检索、可增量更新的 Playbook，而不是不断重写一段总结。

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
