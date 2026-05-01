"""
SOP Agent Package (Thesis Elite Version).
全渠道双塔归因模型：多源异构数据融合、多模态证据建模与复合满意度分析引擎。
"""

from .planner import SOPPlanner, PlanStep
from .executor import SOPExecutor
from .memory import AgentMemory, Artifact
from .tools import LocalTools, LocalPaths

__version__ = "4.0.0"
__author__ = "Thesis Researcher"

__all__ = [
    "SOPPlanner",
    "PlanStep",
    "SOPExecutor",
    "AgentMemory",
    "Artifact",
    "LocalTools",
    "LocalPaths"
]