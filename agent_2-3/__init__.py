# -*- coding: utf-8 -*-
"""
medical_data_cleaner_v2 — 基于 ReAct 的医疗数据清洗系统

通过一个统一的 ReActAgent 探索数据、制定处理计划，
并自主调用工具完成 CSV / 图像 / 文本 / 目录的全流程处理。

快速入口：
    from medical_data_cleaner_v2 import process_medical_data
    result = await process_medical_data("/path/to/data", "/path/to/output")
"""

from .agents import ReactMedicalAgent, process_medical_data

__all__ = ["ReactMedicalAgent", "process_medical_data"]
