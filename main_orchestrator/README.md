# Main Orchestrator

`main_orchestrator` 只负责任务分流、主编排、handoff、状态机兜底和主链路记忆接入。Step1 的真实工具和 runtime 保持在 `step-1/`。

## 目录分层

```text
main_orchestrator/
├── main_orchestrator.py      # CLI / 交互入口
├── agents/                   # TaskRouterAgent、OrchestratorAgent、AgentScope runtime helper
├── core/                     # TaskSpec / WorkerResult 契约、确定性 router、validator
├── execution/                # handoff adapter、Step supervisor、legacy state machine
├── memory/                   # Playbook Memory Supervisor 接入
└── tests/                    # 主链路、handoff、router、Step1 runtime 边界测试
```

## 当前边界

- `TaskRouterAgent` 负责把自然语言或路径转成 `TaskSpec`。
- `OrchestratorAgent` 负责根据 `TaskSpec` 调用子 Agent handoff。
- 当前只有 Step1 已接成 ReAct 子智能体；Step2-7 仍待后续以同样方式接入。
- Memory 当前接入的是 Playbook pre-run/post-run，ReMe 仍保持独立 memory_agent 能力，暂不进入主链路。
