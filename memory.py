"""
Research-grade memory store for the SOP agent (Thesis Elite Version).
核心特性：
1. 自动化审计：自动计算数据损耗率（Sample Retention Rate）。
2. 多维产出物：记录 Parquet、Excel、模型及报告的完整生命周期。
3. 论文辅助：支持一键导出《研究审计清单》，直接用于论文的方法论部分。
"""

from __future__ import annotations

import json
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class Artifact:
    """
    存储实验产出物元数据。
    """
    name: str
    path: str
    description: str
    platform: str = "Common"  # taobao / ozon / mixed
    stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class AgentMemory:
    """
    实验全生命周期记忆库：记录从数据融合(S0)到回归分析(S6)的所有关键状态。
    """
    # 实验超参数记录（如 CSI 权重、回归显著性阈值等）
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    # 存储关键科研笔记（如 VIF 警告、收敛异常、异常值处理理由）
    notes: List[Dict[str, Any]] = field(default_factory=list)
    # 存储物理产出物路径及其元数据
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    # 核心审计日志：用于生成论文中的“实验流程表”
    research_log: List[Dict[str, Any]] = field(default_factory=list)
    # 记录某一步实际产出了哪些字段/文件
    step_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def register_step_output(self, step_id: str, info: Dict[str, Any]) -> None:
        self.step_outputs[step_id] = info

    def set_param(self, key: str, value: Any) -> None:
        """记录实验超参数，确保研究可复现。"""
        self.hyperparameters[key] = value

    def add_note(self, step_id: str, content: str) -> None:
        """添加科研笔记。"""
        self.notes.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step_id,
            "note": content
        })

    def add_artifact(self, artifact: Artifact) -> None:
        """归档实验产出物。"""
        self.artifacts[artifact.name] = artifact

    def log_step(
            self,
            step_id: str,
            action: str,
            input_count: int,
            output_count: int,
            detail: str = "",
            platform: str = "Common",
            artifact_name: str = ""
    ) -> None:
        """
        [科研必备] 记录数据流向损耗，自动生成留存率。
        """
        retention_rate = (output_count / input_count) if input_count > 0 else 1.0

        entry = {
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Step_ID": step_id,
            "Action": action,
            "Platform": platform,
            "Artifact_Name": artifact_name,
            "Rows_In": input_count,
            "Rows_Out": output_count,
            "Retention_Rate": f"{retention_rate:.2%}",
            "Observation": detail
        }
        self.research_log.append(entry)

    def get_artifact_path(self, name: str) -> Optional[str]:
        """获取指定产出物的物理路径。"""
        art = self.artifacts.get(name)
        return art.path if art else None

    def export_audit_report(self, output_dir: Path) -> str:
        """
        [论文必产] 导出 Excel 格式的《研究审计清单》。
        直接用于论文中“数据流向与预处理”部分的描述。
        """
        manifest_path = output_dir / "Research_Audit_Manifest.xlsx"

        # 1. 导出执行轨迹
        df_log = pd.DataFrame(self.research_log)

        # 2. 导出产出物清单
        artifacts_data = []
        for name, art in self.artifacts.items():
            artifacts_data.append({
                "Artifact_Name": name,
                "Platform": art.platform,
                "Path": art.path,
                "Description": art.description,
                "Created_At": art.timestamp
            })
        df_artifacts = pd.DataFrame(artifacts_data)

        # 写入 Excel 不同 Sheet
        with pd.ExcelWriter(manifest_path, engine='xlsxwriter') as writer:
            df_log.to_excel(writer, sheet_name='Execution_Audit', index=False)
            df_artifacts.to_excel(writer, sheet_name='Artifacts_Inventory', index=False)

            # 如果有超参数，也导出一页
            if self.hyperparameters:
                df_params = pd.DataFrame(list(self.hyperparameters.items()), columns=['Parameter', 'Value'])
                df_params.to_excel(writer, sheet_name='Hyperparameters', index=False)
            if self.notes:
                df_notes = pd.DataFrame(self.notes)
                df_notes.to_excel(writer, sheet_name='Research_Notes', index=False)
            if self.step_outputs:
                step_output_rows = []
                for step_id, info in self.step_outputs.items():
                    row = {"Step_ID": step_id}
                    row.update(info)
                    step_output_rows.append(row)
                df_step_outputs = pd.DataFrame(step_output_rows)
                df_step_outputs.to_excel(writer, sheet_name='Step_Outputs', index=False)

        return str(manifest_path)

    def save_research_manifest(self, output_path: str) -> None:
        """
        将整个实验状态持久化为 JSON。
        """
        manifest = {
            "hyperparameters": self.hyperparameters,
            "research_notes": self.notes,
            "execution_trace": self.research_log,
            "step_outputs": self.step_outputs,
            "produced_artifacts": {k: v.__dict__ for k, v in self.artifacts.items()}
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)