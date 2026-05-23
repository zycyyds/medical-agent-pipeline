#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容旧入口：Agent2-3 新主入口是 main_v2.py。"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("main_v2.py")
    spec = importlib.util.spec_from_file_location("agent2_3_main_v2", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 Agent2-3 新入口: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
