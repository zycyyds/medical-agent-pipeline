"""Autonomous medical data pipeline.

This package is intentionally independent from the legacy agent_* / step-* /
main_orchestrator modules. Logic migrated from the legacy pipeline must live
inside this package before it is imported here.
"""

__all__ = ["config", "contracts", "agent_factory"]
