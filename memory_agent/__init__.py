"""MemoryAgent ReMe service package."""

__version__ = "5.0.0"

from .services import MemoryService, get_default_memory_service

__all__ = [
    "MemoryService",
    "get_default_memory_service",
]
