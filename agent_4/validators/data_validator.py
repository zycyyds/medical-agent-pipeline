# validators/data_validator.py
import pandas as pd
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import json
import os


class DataValidator:
    """
    数据验证器，用于在清洗完成后验证数据质量和完整性
    """

    def __init__(self, df: pd.DataFrame, config_path: Optional[str] = None):
        self.df = df
        self.config = self._load_config(config_path) if config_path else {}

    def _load_config(self, config_path: str) -> Dict:
        """加载验证配置"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def validate_column_completeness(self) -> Dict[str, float]:
        """
        验证列的完整性（缺失值比例）
        """
        completeness = {}
        total_rows = len(self.df)

        for col in self.df.columns:
            non_null_count = self.df[col].notna().sum()
            completeness[col] = non_null_count / total_rows if total_rows > 0 else 0

        return completeness

    def validate_data_types(self) -> Dict[str, str]:
        """
        验证数据类型的一致性
        """
        type_consistency = {}

        for col in self.df.columns:
            # 尝试推断数据类型
            sample_non_null = self.df[col].dropna().head(100)  # 取样以提高性能

            if len(sample_non_null) == 0:
                type_consistency[col] = "empty"
                continue

            # 检查是否都是数值
            numeric_mask = pd.to_numeric(sample_non_null, errors='coerce').notna()
            if numeric_mask.all():
                type_consistency[col] = "numeric"
                continue

            # 检查是否都是日期
            date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y年%m月%d日']
            is_date = False
            for fmt in date_formats:
                try:
                    pd.to_datetime(sample_non_null, format=fmt, errors='raise')
                    type_consistency[col] = f"date_format_{fmt}"
                    is_date = True
                    break
                except:
                    continue
            if is_date:
                continue

            # 其他情况归为文本
            type_consistency[col] = "text"

        return type_consistency

    def detect_outliers(self, threshold: float = 3.0) -> Dict[str, List[int]]:
        """
        检测数值列的异常值（使用Z-score方法）
        """
        outliers = {}

        for col in self.df.columns:
            numeric_series = pd.to_numeric(self.df[col], errors='coerce')
            if numeric_series.notna().any():
                # 计算Z-score
                mean_val = numeric_series.mean()
                std_val = numeric_series.std()

                if std_val != 0:  # 避免除零
                    z_scores = abs((numeric_series - mean_val) / std_val)
                    outlier_indices = z_scores[z_scores > threshold].index.tolist()
                    outliers[col] = outlier_indices

        return outliers

    def validate_range_constraints(self, constraints: Optional[Dict] = None) -> Dict[str, List[int]]:
        """
        验证数据范围约束（如年龄应在0-150之间）
        """
        if constraints is None:
            constraints = self.config.get('constraints', {})

        violations = {}

        for col, constraint in constraints.items():
            if col in self.df.columns:
                violations[col] = []

                # 数值范围检查
                if 'min' in constraint or 'max' in constraint:
                    numeric_series = pd.to_numeric(self.df[col], errors='coerce')

                    if 'min' in constraint:
                        min_violations = self.df[numeric_series < constraint['min']].index.tolist()
                        violations[col].extend(min_violations)

                    if 'max' in constraint:
                        max_violations = self.df[numeric_series > constraint['max']].index.tolist()
                        violations[col].extend(max_violations)

                # 正则表达式模式检查
                if 'pattern' in constraint:
                    pattern_violations = self.df[
                        ~self.df[col].astype(str).str.contains(constraint['pattern'], regex=True, na=False)
                    ].index.tolist()
                    violations[col].extend(pattern_violations)

        # 去重并按索引排序
        for col in violations:
            violations[col] = sorted(list(set(violations[col])))

        return violations

    def get_unused_constraints(self, constraints: Optional[Dict] = None) -> List[str]:
        """
        返回配置中声明但当前数据集中不存在的约束列。
        """
        if constraints is None:
            constraints = self.config.get('constraints', {})
        return sorted([col for col in constraints if col not in self.df.columns])

    def run_comprehensive_validation(self, custom_constraints: Optional[Dict] = None) -> Dict:
        """
        运行全面的数据验证
        """
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'basic_stats': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'duplicate_rows': int(self.df.duplicated().sum())
            },
            'completeness': self.validate_column_completeness(),
            'data_types': self.validate_data_types(),
            'outliers': self.detect_outliers(),
            'range_violations': self.validate_range_constraints(custom_constraints),
            'unused_constraints': self.get_unused_constraints(custom_constraints),
        }

        return validation_report

    def generate_quality_score(self, weights: Optional[Dict] = None) -> float:
        """
        生成数据质量评分
        """
        if weights is None:
            weights = {
                'completeness': 0.4,
                'validity': 0.3,
                'consistency': 0.2,
                'accuracy': 0.1
            }

        # 计算各项指标
        completeness_avg = np.mean(list(self.validate_column_completeness().values()))

        # 简化的有效性评估（这里可以根据实际需求定制更复杂的逻辑）
        outliers_count = sum(len(indices) for indices in self.detect_outliers().values())
        validity_score = 1.0 - (outliers_count / len(self.df)) if len(self.df) > 0 else 1.0

        # 检查范围违规数量
        range_violations = self.validate_range_constraints()
        violations_count = sum(len(indices) for indices in range_violations.values())
        validity_score = max(validity_score - (violations_count / len(self.df)), 0) if len(self.df) > 0 else 1.0

        # 一致性和准确性（这里只是示意）
        consistency_score = 0.95  # 假设经过清洗后的数据一致性较高
        accuracy_score = 0.90     # 假设准确性也比较高

        quality_score = (
            completeness_avg * weights['completeness'] +
            validity_score * weights['validity'] +
            consistency_score * weights['consistency'] +
            accuracy_score * weights['accuracy']
        )

        return min(quality_score, 1.0)  # 限制在0-1范围内

    def generate_detailed_report(self) -> str:
        """
        生成详细的人类可读报告
        """
        report_parts = []
        report_parts.append("=== 数据质量验证报告 ===")
        report_parts.append(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append(f"总行数: {len(self.df)}")
        report_parts.append(f"总列数: {len(self.df.columns)}")

        # 缺失值统计
        completeness = self.validate_column_completeness()
        low_completeness = {k: v for k, v in completeness.items() if v < 0.8}  # 低于80%完整性

        if low_completeness:
            report_parts.append("\n⚠️  低完整性列 (完整性 < 80%):")
            for col, comp in low_completeness.items():
                report_parts.append(f"  - {col}: {comp:.2%}")

        # 异常值统计
        outliers = self.detect_outliers()
        outlier_cols = {k: len(v) for k, v in outliers.items() if len(v) > 0}

        if outlier_cols:
            report_parts.append("\n🔍 发现异常值的列:")
            for col, count in outlier_cols.items():
                report_parts.append(f"  - {col}: {count} 个异常值")

        # 范围违规统计
        violations = self.validate_range_constraints()
        violation_cols = {k: len(v) for k, v in violations.items() if len(v) > 0}
        unused_constraints = self.get_unused_constraints()

        if violation_cols:
            report_parts.append("\n❌ 范围违规的列:")
            for col, count in violation_cols.items():
                report_parts.append(f"  - {col}: {count} 个违规项")

        if unused_constraints:
            report_parts.append("\nℹ️ 未命中当前数据列的配置约束:")
            for col in unused_constraints:
                report_parts.append(f"  - {col}")

        # 数据类型统计
        data_types = self.validate_data_types()
        type_summary = {}
        for dt in data_types.values():
            type_summary[dt] = type_summary.get(dt, 0) + 1

        report_parts.append("\n📊 数据类型分布:")
        for dt, count in type_summary.items():
            report_parts.append(f"  - {dt}: {count} 列")

        # 质量评分
        quality_score = self.generate_quality_score()
        report_parts.append(f"\n📈 整体质量评分: {quality_score:.2f}")

        if quality_score >= 0.9:
            report_parts.append("✅ 数据质量优秀")
        elif quality_score >= 0.8:
            report_parts.append("✅ 数据质量良好")
        elif quality_score >= 0.6:
            report_parts.append("⚠️  数据质量一般，建议检查")
        else:
            report_parts.append("❌ 数据质量较差，需要改进")

        return "\n".join(report_parts)
