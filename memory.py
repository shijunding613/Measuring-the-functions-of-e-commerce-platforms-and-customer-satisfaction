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
    platform: str = "Common"
    stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class AgentMemory:
    """
    实验全生命周期记忆库：记录从数据融合(S0)到回归分析(S6)的所有关键状态。
    """

    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    research_log: List[Dict[str, Any]] = field(default_factory=list)
    step_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    completed_steps: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def register_step_output(self, step_id: str, info: Dict[str, Any]) -> None:
        self.step_outputs[step_id] = info

    def mark_step_completed(self, step_id: str, output_path: str, rows: int = 0) -> None:
        self.completed_steps[step_id] = {
            "output_path": output_path,
            "rows": rows,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def is_step_completed(self, step_id: str) -> bool:
        return step_id in self.completed_steps

    def set_param(self, key: str, value: Any) -> None:
        self.hyperparameters[key] = value

    def add_note(self, step_id: str, content: str) -> None:
        self.notes.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step_id,
            "note": content
        })

    def add_artifact(self, artifact: Artifact) -> None:
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
        retention_rate = (output_count / input_count) if input_count > 0 else 1.0

        entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        art = self.artifacts.get(name)
        return art.path if art else None

    def export_audit_report(self, output_dir: Path) -> str:
        """
        导出 Excel 格式的《研究审计清单》。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "Research_Audit_Manifest.xlsx"

        df_log = pd.DataFrame(
            self.research_log,
            columns=[
                "Timestamp", "Step_ID", "Action", "Platform",
                "Artifact_Name", "Rows_In", "Rows_Out",
                "Retention_Rate", "Observation"
            ]
        )

        artifacts_data = []
        for name, art in self.artifacts.items():
            artifacts_data.append({
                "Artifact_Name": name,
                "Platform": art.platform,
                "Path": art.path,
                "Description": art.description,
                "Created_At": art.timestamp,
                "Stats": json.dumps(self._make_json_safe(art.stats), ensure_ascii=False)
            })

        df_artifacts = pd.DataFrame(
            artifacts_data,
            columns=[
                "Artifact_Name", "Platform", "Path",
                "Description", "Created_At", "Stats"
            ]
        )

        with pd.ExcelWriter(manifest_path, engine="openpyxl") as writer:
            df_log.to_excel(writer, sheet_name="Execution_Audit", index=False)
            df_artifacts.to_excel(writer, sheet_name="Artifacts_Inventory", index=False)

            df_params = pd.DataFrame(
                list(self.hyperparameters.items()),
                columns=["Parameter", "Value"]
            )
            df_params.to_excel(writer, sheet_name="Hyperparameters", index=False)

            df_notes = pd.DataFrame(
                self.notes,
                columns=["timestamp", "step", "note"]
            )
            df_notes.to_excel(writer, sheet_name="Research_Notes", index=False)

            step_output_rows = []
            for step_id, info in self.step_outputs.items():
                row = {"Step_ID": step_id}

                for k, v in info.items():
                    safe_v = self._make_json_safe(v)

                    if isinstance(safe_v, (dict, list)):
                        safe_v = json.dumps(safe_v, ensure_ascii=False)

                    row[k] = safe_v

                step_output_rows.append(row)

            df_step_outputs = pd.DataFrame(step_output_rows)
            df_step_outputs.to_excel(writer, sheet_name="Step_Outputs", index=False)

            completed_rows = []
            for step_id, info in self.completed_steps.items():
                row = {"Step_ID": step_id}
                row.update(info)
                completed_rows.append(row)

            df_completed = pd.DataFrame(completed_rows)
            df_completed.to_excel(writer, sheet_name="Completed_Steps", index=False)

        return str(manifest_path)

    def save_research_manifest(self, output_path: str) -> None:
        """
        将整个实验状态持久化为 JSON。
        """
        manifest = {
            "completed_steps": self.completed_steps,
            "hyperparameters": self.hyperparameters,
            "research_notes": self.notes,
            "execution_trace": self.research_log,
            "step_outputs": self.step_outputs,
            "produced_artifacts": {
                k: {
                    "name": v.name,
                    "path": v.path,
                    "description": v.description,
                    "platform": v.platform,
                    "stats": v.stats,
                    "timestamp": v.timestamp
                }
                for k, v in self.artifacts.items()
            }
        }

        safe_manifest = self._make_json_safe(manifest)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(safe_manifest, f, indent=4, ensure_ascii=False)

    def _make_json_safe(self, obj: Any) -> Any:
        """
        防止 Path、numpy、pandas、datetime、dict/list 等类型导致 JSON 或 Excel 保存失败。
        """
        if obj is None:
            return None

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(obj, dict):
            return {str(k): self._make_json_safe(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._make_json_safe(x) for x in obj]

        if isinstance(obj, tuple):
            return [self._make_json_safe(x) for x in obj]

        # pandas / numpy 标量
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass

        # pandas Timestamp
        if hasattr(obj, "strftime"):
            try:
                return obj.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        # NaN / NA 安全判断
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass

        return obj