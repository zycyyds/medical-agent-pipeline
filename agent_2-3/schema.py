# -*- coding: utf-8 -*-
"""Medical Information Extraction Schema using Pydantic."""
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field


class MedicalEntity(BaseModel):
    """Schema for a medical entity extracted from medical records."""
    name: str = Field(
        ..., 
        description=(
            "医学实体的标准名称，根据category不同有不同要求：\n"
            "- Test: 检验项目全称，如'类风湿因子'(非'风湿因子')、'白细胞计数'(非'白细胞')\n"
            "- Disease: 疾病规范名称，如'2型糖尿病'、'原发性高血压'\n"
            "- Drug: 药物通用名/商品名，如'阿司匹林'、'二甲双胍'\n"
            "- Symptom: 症状名称，如'头痛'、'发热'、'恶心'\n"
            "- Treatment: 治疗/手术名称，如'冠状动脉搭桥术'\n"
            "- Anatomy: 解剖部位，如'左侧颞叶'、'右肺下叶'"
        )
    )
    category: Literal[
        "Disease", "Drug", "Symptom", "Test", "Treatment", "Anatomy", "LabValue", "Finding", "Other"
    ] = Field(
        ..., 
        description=(
            "实体类别：\n"
            "- Test: 检验项目（血常规、生化、免疫等）\n"
            "- Disease: 疾病/诊断\n"
            "- Drug: 药物/药品\n"
            "- Symptom: 症状/主诉\n"
            "- Treatment: 治疗/手术/操作\n"
            "- Anatomy: 解剖部位\n"
            "- LabValue: 其他定量指标\n"
            "- Finding: 影像学/检查发现（如无实变、无积液）\n"
            "- Other: 其他无法归类的医学实体"
        )
    )
    original_text: Optional[str] = Field(
        None, 
        description="从原文中提取的文本片段，用于溯源（可选）"
    )
    value: Optional[Union[str, int, float]] = Field(
        None, 
        description=(
            "实体的值，根据category不同：\n"
            "- Test: 检验结果（数值如20.0，或定性如'阴性'）\n"
            "- Disease: 分期/分级（如'III期'、'中度'）\n"
            "- Drug: 剂量（如'100mg'）\n"
            "- Symptom: 程度/持续时间（如'剧烈'、'3天'）"
        )
    )
    unit: Optional[str] = Field(
        None,
        description=(
            "单位或用法：\n"
            "- Test: 检验单位（10^9/L、g/L、IU/mL、%等）\n"
            "- Drug: 用法（每日一次、口服等）"
        )
    )


class TemporalAttribute(BaseModel):
    """Schema for temporal information."""
    event: str = Field(..., description="The event associated with this time (e.g., onset, admission).")
    time_expression: str = Field(..., description="The temporal expression found in text.")
    normalized_time: Optional[str] = Field(None, description="Normalized time if possible (e.g., YYYY-MM-DD).")


class QuantityAttribute(BaseModel):
    """Schema for quantity/dosage information."""
    drug_or_treatment: str = Field(..., description="The drug or treatment this applies to.")
    amount: Optional[str] = Field(None, description="Dosage amount (e.g., 500mg).")
    frequency: Optional[str] = Field(None, description="Frequency (e.g., twice a day).")
    duration: Optional[str] = Field(None, description="Duration (e.g., for 5 days).")


class MedicalRelation(BaseModel):
    """Schema for relations between entities."""
    source: str = Field(..., description="The source entity name.")
    target: str = Field(..., description="The target entity name.")
    relation_type: Literal[
        "Causal", "Temporal", "TreatmentFor", "ManifestationOf", "LocatedIn", "TestResult"
    ] = Field(..., description="The type of relation.")


class MedicalExtractionResult(BaseModel):
    """The complete structured result for medical information extraction."""
    entities: List[MedicalEntity] = Field(default_factory=list, description="List of extracted entities.")
    temporal_info: List[TemporalAttribute] = Field(default_factory=list, description="List of temporal attributes.")
    quantity_info: List[QuantityAttribute] = Field(default_factory=list, description="List of dosage/quantity attributes.")
    relations: List[MedicalRelation] = Field(default_factory=list, description="List of relations between entities.")
    impression: Optional[str] = Field(None, description="诊断印象/结论（如'无急性心肺病变'）")
    indication: Optional[str] = Field(None, description="检查指征/目的（如'评估感染'、'胸痛待查'）")