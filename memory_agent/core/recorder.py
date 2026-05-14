"""
执行记录器 (Execution Recorder)

负责：
1. 从 First Agent 执行中提取轨迹信息
2. 生成执行指纹用于去重
3. 提取策略描述
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExecutionTrace:
    """执行轨迹数据类"""

    # 输入输出
    input: str  # 用户输入
    output: str  # Agent 输出

    # 工具调用信息
    tool_sequence: list[str] = field(default_factory=list)  # 工具调用序列
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # 详细调用信息

    # 执行结果
    success: bool = True  # 是否成功
    result: str = ""  # 结果描述

    # 性能信息
    duration: float = 0.0  # 执行时间 (秒)
    start_time: datetime = field(default_factory=datetime.now)

    # 派生信息
    input_pattern: str = ""  # 输入模式（用于指纹）

    def __post_init__(self):
        """初始化后处理"""
        if not self.input_pattern:
            self.input_pattern = self._extract_input_pattern()
        if not self.result:
            self.result = self.output

    def _extract_input_pattern(self) -> str:
        """提取输入模式"""
        # 简化输入，提取关键模式
        # 例如：目录路径 -> "目录：xxx"
        # 例如：文件路径 -> "文件：xxx.jpg"
        input_lower = self.input.lower()

        if any(ext in input_lower for ext in [".jpg", ".jpeg", ".png", ".pdf"]):
            return f"文件：{self.input.split('/')[-1]}"

        if "目录" in self.input or "文件夹" in self.input:
            parts = self.input.split("/")
            return f"目录：{parts[-1] if parts[-1] else parts[-2]}"

        return self.input[:50]  # 默认截取前 50 字符


class ExecutionRecorder:
    """执行记录器"""

    def __init__(self):
        self.hasher = hashlib.sha256()

    def generate_fingerprint(self, trace: ExecutionTrace) -> str:
        """
        生成执行指纹用于去重

        指纹 = hash(输入模式 + 工具调用序列)

        Args:
            trace: 执行轨迹

        Returns:
            SHA256 指纹字符串
        """
        # 构建指纹键
        tool_seq_str = "|".join(trace.tool_sequence)
        key = f"{trace.input_pattern}:{tool_seq_str}"

        # 生成哈希
        h = hashlib.sha256(key.encode())
        return h.hexdigest()

    def extract_strategy_description(self, trace: ExecutionTrace) -> str:
        """
        从执行轨迹中提取策略描述

        Args:
            trace: 执行轨迹

        Returns:
            策略描述字符串
        """
        if trace.success:
            # 成功策略
            tools_used = ", ".join(trace.tool_sequence[:3])  # 最多显示 3 个工具
            return f"使用 {tools_used} 成功处理 {trace.input_pattern}"
        else:
            # 失败模式
            return f"处理 {trace.input_pattern} 时失败：{trace.output[:100]}"

    def classify_execution_type(self, trace: ExecutionTrace) -> str:
        """
        分类执行类型

        Args:
            trace: 执行轨迹

        Returns:
            类型标签
        """
        if trace.success:
            # 检查是否包含特定模式
            input_lower = trace.input.lower()

            if any(kw in input_lower for kw in ["眼位", "eye", "ocular"]):
                return "domain_heuristic"
            if len(trace.tool_sequence) > 3:
                return "success_strategy"

            return "success_strategy"
        else:
            return "failure_mode"

    def trace_to_dict(self, trace: ExecutionTrace) -> dict:
        """将轨迹转换为字典"""
        return {
            "input": trace.input,
            "output": trace.output,
            "tool_sequence": trace.tool_sequence,
            "success": trace.success,
            "duration": trace.duration,
            "input_pattern": trace.input_pattern,
            "start_time": trace.start_time.isoformat(),
        }
