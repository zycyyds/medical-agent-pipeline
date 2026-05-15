from __future__ import annotations

import sys
from pathlib import Path

AGENT1_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT1_DIR.parent
STEP1_DIR = PROJECT_ROOT / "step-1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STEP1_DIR) in sys.path:
    sys.path.remove(str(STEP1_DIR))
sys.path.insert(0, str(STEP1_DIR))

from step1_tools_core import *  # noqa: F401,F403
