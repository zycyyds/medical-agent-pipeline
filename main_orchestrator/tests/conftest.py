import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
for path in (
    PROJECT_ROOT,
    ORCHESTRATOR_DIR,
    ORCHESTRATOR_DIR / "agents",
    ORCHESTRATOR_DIR / "core",
    ORCHESTRATOR_DIR / "execution",
    ORCHESTRATOR_DIR / "memory",
    PROJECT_ROOT / "step-1",
    PROJECT_ROOT / "step-4",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
