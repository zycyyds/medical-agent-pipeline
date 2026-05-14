"""
MLDataProcessorAgent — 基于 AgentScope 框架的机器学习数据预处理 Agent

功能：
  1. 加载原始 CSV（ml_dataset_*.csv）
  2. 丢弃 sample_row_id、目标列（y / label_name 在输出末尾附上）；保留 patient_id（原样、不编码）
  3. 删除「初诊」占位列：名为 Table_初诊 及其重复列（Table_初诊.1 等）或以 Table_初诊-/Table_初诊_ 开头的列
  4. 诊断及 ICD 相关列：默认原样写入 CSV，不参与编码与数值填充
  5. 处理 Extracted_LabValue_* 系列：保留数值版本（_value），丢弃文本版本
  6. 其余数值列：转数字、缺失用中位数填充（不标准化）
  7. 其余非数值列：LabelEncoder（NaN 视为「缺失」）
  8. 输出: dataset_processed.csv、preprocessing_meta.json

用法:
    python ml_data_processor_agent.py  [可选: --input <path> --output_dir <dir>]
"""

import os
import sys
import json
import argparse

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── AgentScope 兼容层 ─────────────────────────────────────────────────────────
try:
    import agentscope
    from agentscope.agents import AgentBase
    from agentscope.message import Msg
    HAS_AGENTSCOPE = True
except ImportError:
    HAS_AGENTSCOPE = False

    class Msg:
        """AgentScope Msg 的最小兼容实现。"""
        def __init__(self, name: str, content, role: str = "assistant"):
            self.name = name
            self.content = content
            self.role = role

        def __repr__(self):
            return f"Msg(name={self.name!r}, role={self.role!r}, content={self.content!r})"

    class AgentBase:
        """AgentScope AgentBase 的最小兼容实现。"""
        def __init__(self, name: str, **kwargs):
            self.name = name

        def reply(self, x: "Msg" = None) -> "Msg":
            raise NotImplementedError


# ── 数据预处理 Agent ──────────────────────────────────────────────────────────
class MLDataProcessorAgent(AgentBase):
    """
    前庭疾病数据 → ML 特征矩阵预处理 Agent。

    Parameters
    ----------
    name        : Agent 名称
    output_dir  : 输出目录（默认与输入文件同目录）
    drop_leaky  : 为 True 时从输出中完全移除诊断/ICD 列（严格建模）；默认 False 保留原值
    """

    # 诊断结论、ICD10 等：默认原样保留；drop_leaky=True 时整列删除且不参与建模特征
    _LEAKY_COLS = [
        # 初步诊断 Table 列
        "Table_初步诊断-初步诊断-综合征类型",
        "Table_初步诊断-初步诊断-定位诊断",
        "Table_初步诊断-初步诊断-可能诊断4",
        "Table_初步诊断-初步诊断-备注诊断",
        "Table_初步诊断-初步诊断-可能诊断3",
        # 展开后的疾病名称 / ICD10 列
        "综合征类型_疾病名称", "综合征类型_ICD10",
        "定位诊断_疾病名称",   "定位诊断_ICD10",
        "可能诊断4_疾病名称",  "可能诊断4_ICD10",
        "备注诊断_疾病名称",   "备注诊断_ICD10",
        "可能诊断3_疾病名称",  "可能诊断3_ICD10",
        "label_ICD10",
    ]

    # 仅去掉行序号；patient_id 保留在输出中且不参与编码
    _ID_DROP_COLS = ["sample_row_id"]
    _PASSTHROUGH_ID_COLS = ["patient_id"]
    _TARGET_COLS = ["y", "label_name"]

    def __init__(
        self,
        name: str = "MLDataProcessorAgent",
        output_dir: str = None,
        drop_leaky: bool = False,
    ):
        super().__init__(name=name)
        self.output_dir = output_dir
        self.drop_leaky = drop_leaky

    # ── 对外接口 ──────────────────────────────────────────────────────────────
    def reply(self, x: Msg = None) -> Msg:
        """
        接受一条消息，内容为文件路径字符串或 dict（含 'file_path' 键）。
        返回包含处理摘要的 Msg。
        """
        if x is None or not x.content:
            return Msg(self.name, {"status": "error", "message": "未收到输入"}, role="assistant")

        content = x.content
        if isinstance(content, str):
            file_path = content
        elif isinstance(content, dict):
            file_path = content.get("file_path", "")
        else:
            return Msg(self.name, {"status": "error", "message": "输入格式不正确"}, role="assistant")

        self._log(f"开始处理: {file_path}")
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")  # 自动去除 BOM
            result = self._process(df, file_path)
            return Msg(self.name, result, role="assistant")
        except Exception as e:
            import traceback
            return Msg(self.name, {"status": "error", "message": str(e),
                                   "traceback": traceback.format_exc()}, role="assistant")

    # ── 内部处理流程 ──────────────────────────────────────────────────────────
    def _process(self, df: pd.DataFrame, source_path: str) -> dict:
        self._log(f"原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")

        # ① 提取目标变量
        y = df["y"].astype(int).reset_index(drop=True)
        label_name = df["label_name"].reset_index(drop=True)
        self._log(f"标签分布: {y.value_counts().to_dict()}")

        # ② 确定要删除的列（不含 patient_id）
        drop_cols = list(self._ID_DROP_COLS) + list(self._TARGET_COLS)
        if self.drop_leaky:
            drop_cols += list(self._LEAKY_COLS)
        drop_cols = [c for c in drop_cols if c in df.columns]
        df_feat = df.drop(columns=drop_cols).copy()

        # ③ 去掉「Table_初诊」类占位列（常见于重复表头），保留其后辅助检查等列
        chuzhen_drop = [c for c in df_feat.columns if self._is_chuzhen_junk_col(c)]
        if chuzhen_drop:
            df_feat.drop(columns=chuzhen_drop, inplace=True)
            self._log(f"删除初诊占位列 {len(chuzhen_drop)} 个: {chuzhen_drop}")

        # ④ 删除 Extracted_LabValue 中的原始文本列，仅保留 _value 数值列
        text_lab_cols = [
            c for c in df_feat.columns
            if c.startswith("Extracted_LabValue_") and not c.endswith("_value")
        ]
        df_feat.drop(columns=text_lab_cols, inplace=True)
        self._log(f"删除 {len(drop_cols)} 个行序号/目标列，"
                  f"{len(text_lab_cols)} 个检验值文本列")
        self._log(f"剩余特征数: {df_feat.shape[1]}")

        # ⑤ 诊断/ICD 与 patient_id：原样透传，不参与分类与编码
        passthrough_cols = []
        for c in self._PASSTHROUGH_ID_COLS:
            if c in df_feat.columns:
                passthrough_cols.append(c)
        if not self.drop_leaky:
            for c in self._LEAKY_COLS:
                if c in df_feat.columns and c not in passthrough_cols:
                    passthrough_cols.append(c)
        if passthrough_cols:
            self._log(f"原样保留（不编码）列 ({len(passthrough_cols)}): {passthrough_cols}")

        feat_order = list(df_feat.columns)
        passthrough_set = set(passthrough_cols)
        cols_to_process = [c for c in feat_order if c not in passthrough_set]
        df_work = df_feat[cols_to_process].copy()

        # ⑥ 分类列 vs 数值列（仅针对需处理的列）
        numeric_cols, categorical_cols = self._classify_columns(df_work)
        self._log(f"数值列 ({len(numeric_cols)}): {numeric_cols}")
        self._log(f"类别列 ({len(categorical_cols)}): {categorical_cols}")

        # ⑦ 类别变量编码（非数值列 → 整数类别码）
        label_encoders = {}
        df_proc = df_work.copy()
        for col in categorical_cols:
            df_proc[col] = df_proc[col].fillna("缺失").astype(str)
            le = LabelEncoder()
            df_proc[col] = le.fit_transform(df_proc[col])
            label_encoders[col] = le
            self._log(f"  LabelEncoder [{col}]: {list(le.classes_)}")

        # ⑧ 数值列：强制为 float，缺失用列中位数填充（不标准化）
        for col in numeric_cols:
            df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")
            med = df_proc[col].median()
            if pd.isna(med):
                med = 0.0
            df_proc[col] = df_proc[col].fillna(med)

        # ⑨ 低方差列检测（仅针对已处理列，仅提示）
        variances = df_proc.var(numeric_only=True)
        zero_var_cols = variances[variances == 0].index.tolist()
        if zero_var_cols:
            self._log(f"警告: {len(zero_var_cols)} 列方差为 0（全部相同），建议删除: {zero_var_cols}")

        # ⑩ 按原始特征列顺序合并：处理列 + 透传列
        out_parts = []
        for col in feat_order:
            if col in passthrough_set:
                out_parts.append(df_feat[[col]])
            else:
                out_parts.append(df_proc[[col]])
        out_df = pd.concat(out_parts, axis=1)
        out_df["y"] = y.values
        out_df["label_name"] = label_name.values

        # ⑪ 保存输出
        out_dir = self.output_dir or os.path.dirname(os.path.abspath(source_path))
        os.makedirs(out_dir, exist_ok=True)

        dataset_path = os.path.join(out_dir, "dataset_processed.csv")
        meta_path = os.path.join(out_dir, "preprocessing_meta.json")

        out_df.to_csv(dataset_path, index=False, encoding="utf-8-sig")

        feature_cols = list(df_feat.columns)
        meta = {
            "n_samples"           : int(len(out_df)),
            "n_features"          : len(feature_cols),
            "feature_names"       : feature_cols,
            "target_cols"         : ["y", "label_name"],
            "passthrough_raw_cols": passthrough_cols,
            "numeric_cols"        : numeric_cols,
            "categorical_cols"    : categorical_cols,
            "zero_variance_cols"  : zero_var_cols,
            "label_mapping"       : {"0": "单侧原发前庭病变", "1": "良性阵发性位置性眩晕"},
            "label_distribution"  : {str(k): int(v) for k, v in y.value_counts().items()},
            "dropped_cols"        : drop_cols,
            "dropped_chuzhen_cols": chuzhen_drop,
            "dropped_text_lab_cols": text_lab_cols,
            "kept_id_cols"        : [c for c in self._PASSTHROUGH_ID_COLS if c in out_df.columns],
            "label_encoders"      : {col: list(le.classes_)
                                     for col, le in label_encoders.items()},
            "output_files"        : {"dataset": dataset_path, "meta": meta_path},
            "notes"               : "patient_id 与诊断/ICD 见 passthrough_raw_cols，原样保留；已删除 Table_初诊 占位列，见 dropped_chuzhen_cols。",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self._log(f"✓ 数据集已保存: {dataset_path}")
        self._log(f"✓ 元数据已保存: {meta_path}")
        self._log("处理完成！")

        return {
            "status"       : "success",
            "n_samples"    : int(len(out_df)),
            "n_features"   : len(feature_cols),
            "label_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
            "zero_variance_cols": zero_var_cols,
            "output_files" : {"dataset": dataset_path, "meta": meta_path},
        }

    # ── 工具函数 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _is_chuzhen_junk_col(name: str) -> bool:
        """初诊表重复/空表头列，删除后保留后续「辅助检查」等列。"""
        s = str(name).strip()
        if s == "Table_初诊":
            return True
        # pandas 读入重复列名时常见 Table_初诊.1, Table_初诊.2
        if s.startswith("Table_初诊."):
            return True
        if s.startswith("Table_初诊-") or s.startswith("Table_初诊_"):
            return True
        return False

    @staticmethod
    def _classify_columns(df: pd.DataFrame):
        """将 DataFrame 的列分为数值列和类别列。"""
        numeric_cols, categorical_cols = [], []
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            # 若转换后非 NaN 的比例 >= 原非空比例，则认为是数值列
            original_notnull = df[col].notna().sum()
            converted_notnull = converted.notna().sum()
            if original_notnull == 0 or converted_notnull / max(original_notnull, 1) >= 0.9:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        return numeric_cols, categorical_cols

    def _log(self, msg: str):
        print(f"[{self.name}] {msg}", flush=True)


# ── 独立运行入口 ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ML 数据预处理 Agent")
    parser.add_argument(
        "--input", "-i",
        default="output_step7/ml_dataset.csv",
        help="输入 CSV 路径",
    )
    parser.add_argument(
        "--output_dir", "-o",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="输出目录（默认与输入文件同目录）",
    )
    parser.add_argument(
        "--drop_diagnosis", action="store_true",
        help="从输出中移除全部诊断/ICD 列（严格建模、避免标签泄漏时使用）",
    )
    args = parser.parse_args()

    if HAS_AGENTSCOPE:
        agentscope.init(model_configs=[])  # 无需 LLM，仅用 Agent 框架

    agent = MLDataProcessorAgent(
        name="MLDataProcessorAgent",
        output_dir=args.output_dir,
        drop_leaky=args.drop_diagnosis,
    )

    input_msg = Msg(name="user", content=args.input, role="user")
    result_msg = agent.reply(input_msg)

    print("\n" + "=" * 60)
    print("Agent 返回结果：")
    print(json.dumps(result_msg.content, ensure_ascii=False, indent=2))

    # 若 output_dir 未指定，打印预览
    if result_msg.content.get("status") == "success":
        ds_path = result_msg.content["output_files"]["dataset"]
        print("\n处理后数据集预览（前 3 行）：")
        print(pd.read_csv(ds_path).head(3).to_string())


if __name__ == "__main__":
    main()
