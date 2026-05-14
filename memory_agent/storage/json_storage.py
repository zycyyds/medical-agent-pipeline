"""
JSON 文件存储后端

提供记忆库的持久化存储功能。
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonStorage:
    """JSON 文件存储类"""

    def __init__(self, file_path: str | Path):
        """
        初始化 JSON 存储

        Args:
            file_path: JSON 文件路径
        """
        self.file_path = Path(file_path)
        self._data: dict[str, Any] = {}
        self._lock_file = self.file_path.with_suffix(".lock")

    def load(self) -> dict[str, Any]:
        """
        从文件加载数据

        Returns:
            加载的数据字典
        """
        if not self.file_path.exists():
            self._data = self._default_structure()
            self.save()
            return self._data

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告：读取文件失败 {self.file_path}: {e}")
            self._data = self._default_structure()

        return self._data

    def save(self) -> bool:
        """
        保存数据到文件

        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            # 写入临时文件
            temp_path = self.file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2, default=str)

            # 原子替换
            backup_path = None
            if self.file_path.exists():
                backup_path = self.file_path.with_suffix(".bak")
                self.file_path.rename(backup_path)

            temp_path.rename(self.file_path)

            # 清理备份
            if backup_path and backup_path.exists():
                backup_path.unlink()

            return True
        except IOError as e:
            print(f"错误：保存文件失败 {self.file_path}: {e}")
            return False

    def _default_structure(self) -> dict[str, Any]:
        """返回默认的数据结构"""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "entries": [],
            "statistics": {
                "total_executions": 0,
                "unique_strategies": 0,
                "last_reflection": None,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """获取数据"""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置数据"""
        self._data[key] = value

    def append_to_list(self, key: str, item: Any) -> None:
        """追加到列表"""
        if key not in self._data:
            self._data[key] = []
        self._data[key].append(item)

    def update_statistics(self, **kwargs) -> None:
        """更新统计信息"""
        if "statistics" not in self._data:
            self._data["statistics"] = {}
        self._data["statistics"].update(kwargs)

    @property
    def entries(self) -> list[dict]:
        """获取所有条目"""
        return self._data.get("entries", [])

    @entries.setter
    def entries(self, value: list[dict]):
        """设置条目列表"""
        self._data["entries"] = value

    def __len__(self) -> int:
        """返回条目数量"""
        return len(self.entries)
