"""
Executor coordinating the SOP workflow (Thesis Elite Version - Full Cycle).
修复：
1. 统一 Platform 列名
2. 修复 S3 train_dapt 参数缺失问题
3. 对齐 S5.5 / S5.6 多模态流程
4. 保留 S6 回归闭环
"""

from __future__ import annotations
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List

from .memory import AgentMemory, Artifact
from .planner import PlanStep
from .tools import LocalTools


class SOPExecutor:
    def __init__(self, tools: LocalTools, memory: AgentMemory):
        self.tools = tools
        self.memory = memory
        self.logger = logging.getLogger("SOPExecutor")
        self._shared_df = None

        # --- 记录论文核心公式参数 ---
        self.memory.set_param("CSI_Alpha_Coefficient", 0.1)
        self.memory.set_param("Regression_Robust_Type", "HC3")
        self.memory.set_param("VIF_Threshold", 5.0)

    def _check_platform_integrity(
            self,
            df: pd.DataFrame,
            name: str,
            step_id: str,
            expected_platforms: set[str] | None = None,
            strict: bool = False
    ) -> None:
        """
        检查混合数据表是否意外丢失平台。
        strict=True 时，如果期望 Taobao + Ozon 但只剩一个平台，直接报错。
        strict=False 时只写日志和 memory note。
        """
        if df is None or df.empty:
            msg = f"⚠️ [{step_id}] {name} 为空表。"
            self.logger.warning(msg)
            self.memory.add_note(step_id, msg)
            return

        platform_col = "Platform" if "Platform" in df.columns else ("platform" if "platform" in df.columns else None)

        if not platform_col:
            msg = f"⚠️ [{step_id}] {name} 未找到 Platform/platform 列，无法检查平台完整性。"
            self.logger.warning(msg)
            self.memory.add_note(step_id, msg)
            return

        platforms = set(df[platform_col].astype(str).str.lower().dropna().unique().tolist())
        platform_counts = df[platform_col].astype(str).str.lower().value_counts(dropna=False)

        self.logger.info(f"📌 [{step_id}] {name} 平台分布:\n{platform_counts}")

        if expected_platforms:
            missing = expected_platforms - platforms

            if missing:
                msg = (
                    f"⚠️ [{step_id}] {name} 缺失平台: {sorted(missing)}；"
                    f"当前平台: {sorted(platforms)}。"
                    f"如果你预期 Taobao + Ozon 同时存在，这可能是前序节点丢失或覆盖。"
                )
                self.logger.warning(msg)
                self.memory.add_note(step_id, msg)

                if strict:
                    raise ValueError(msg)

    def _save_checkpoint(self, df: pd.DataFrame, name: str, step_id: str) -> str:
        """
        每个关键节点同时保存 parquet + xlsx。
        如果旧文件已经存在，先自动备份到 checkpoints_backup，避免覆盖丢失。

        关键保护：
        1. 每次保存前统计 Platform 分布；
        2. 同时输出平台拆分备份；
        3. 如果混合总表只剩单个平台，写入日志警告，防止 Ozon / Taobao 覆盖式丢失。
        """
        out_dir = self.tools.paths.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        mixed_names = {
            "clean_reviews",
            "featured_metadata",
            "semantic_scored",
            "review_validity_audit",
            "multimodal_base_features",
            "multimodal_evidence",
            "S6_Final_Regression_Input",
        }

        if name in mixed_names:
            self._check_platform_integrity(
                df=df,
                name=name,
                step_id=step_id,
                expected_platforms={"taobao", "ozon"},
                strict = name == "S6_Final_Regression_Input"
            )

        backup_dir = out_dir / "checkpoints_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)

        split_dir = out_dir / "platform_split_checkpoints"
        split_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = out_dir / f"{name}.parquet"
        excel_path = out_dir / f"{name}.xlsx"

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # =========================================================
        # 1. 平台完整性检查：防止 Taobao / Ozon 被覆盖成单平台
        # =========================================================
        platform_counts_dict = {}

        platform_col = None
        if "Platform" in df.columns:
            platform_col = "Platform"
        elif "platform" in df.columns:
            platform_col = "platform"

        if platform_col:
            platform_counts = df[platform_col].astype(str).str.lower().value_counts(dropna=False)
            platform_counts_dict = platform_counts.to_dict()

            self.logger.info(f"📌 [{step_id}] {name} 平台分布:\n{platform_counts}")

            # 只要是混合主流程关键表，正常应该至少包含 taobao / ozon 中的一个或两个
            existing_platforms = set(platform_counts.index.astype(str))

            if name in [
                "clean_reviews",
                "featured_metadata",
                "semantic_scored",
                "review_validity_audit",
                "multimodal_base_features",
                "multimodal_evidence",
                "S6_Final_Regression_Input",
            ]:
                if len(existing_platforms) == 1:
                    only_platform = list(existing_platforms)[0]
                    warning_msg = (
                        f"⚠️ [{step_id}] 警告：当前 checkpoint '{name}' 只包含一个平台：{only_platform}。"
                        f"如果你预期 Taobao + Ozon 同时存在，这可能意味着另一个平台在前序步骤中丢失。"
                    )
                    self.logger.warning(warning_msg)
                    self.memory.add_note(step_id, warning_msg)

            # 每次都额外保存平台拆分版本，防止一个平台覆盖另一个平台
            for platform_value, sub_df in df.groupby(platform_col):
                platform_safe = str(platform_value).lower().replace("/", "_").replace("\\", "_")
                split_parquet = split_dir / f"{name}_{platform_safe}_{step_id}_{timestamp}.parquet"
                split_excel = split_dir / f"{name}_{platform_safe}_{step_id}_{timestamp}.xlsx"

                sub_df.to_parquet(split_parquet, index=False)
                sub_df.to_excel(split_excel, index=False)

                self.logger.info(
                    f"🧩 [{step_id}] 平台拆分备份已保存: {platform_safe} | rows={len(sub_df)}"
                )

        else:
            self.logger.warning(f"⚠️ [{step_id}] {name} 未找到 Platform/platform 列，无法做平台完整性检查。")

        # =========================================================
        # 2. 覆盖前自动备份旧文件
        # =========================================================
        if parquet_path.exists():
            backup_parquet = backup_dir / f"{name}_{step_id}_{timestamp}.parquet"
            parquet_path.replace(backup_parquet)
            self.logger.info(f"🧷 已备份旧 parquet: {backup_parquet}")

        if excel_path.exists():
            backup_excel = backup_dir / f"{name}_{step_id}_{timestamp}.xlsx"
            excel_path.replace(backup_excel)
            self.logger.info(f"🧷 已备份旧 Excel: {backup_excel}")

        # =========================================================
        # 3. 保存新的混合总表 checkpoint
        # =========================================================
        df.to_parquet(parquet_path, index=False)
        df.to_excel(excel_path, index=False)

        missing_cells = int(df.isna().sum().sum())
        total_cells = int(df.shape[0] * df.shape[1])

        self.memory.register_step_output(step_id, {
            "parquet": str(parquet_path),
            "excel": str(excel_path),
            "backup_dir": str(backup_dir),
            "split_dir": str(split_dir),
            "rows": len(df),
            "columns": len(df.columns),
            "platform_counts": platform_counts_dict,
            "column_names": ", ".join(df.columns.astype(str).tolist()),
            "missing_cells": missing_cells,
            "missing_rate": round(float(missing_cells / total_cells), 6) if total_cells > 0 else 0,
        })

        self.memory.mark_step_completed(step_id, str(parquet_path), rows=len(df))

        self.logger.info(f"💾 checkpoint 已保存: {parquet_path}")
        self.logger.info(f"💾 Excel 备份已保存: {excel_path}")

        return str(parquet_path)

    def _load_checkpoint(self, name: str) -> pd.DataFrame | None:
        """
        优先读取 parquet checkpoint。
        读取后立即检查 Platform 分布，防止加载到只剩单平台的旧错误结果。
        """
        parquet_path = self.tools.paths.output_dir / f"{name}.parquet"

        if not parquet_path.exists():
            return None

        self.logger.info(f"✅ 发现 checkpoint，直接读取: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        platform_col = None
        if "Platform" in df.columns:
            platform_col = "Platform"
        elif "platform" in df.columns:
            platform_col = "platform"

        if platform_col:
            platform_counts = df[platform_col].astype(str).str.lower().value_counts(dropna=False)
            self.logger.info(f"📌 [LOAD] {name} checkpoint 平台分布:\n{platform_counts}")

            critical_names = [
                "clean_reviews",
                "featured_metadata",
                "semantic_scored",
                "review_validity_audit",
                "multimodal_base_features",
                "multimodal_evidence",
                "S6_Final_Regression_Input",
            ]

            if name in critical_names and len(platform_counts.index) == 1:
                only_platform = str(platform_counts.index[0])
                warning_msg = (
                    f"⚠️ [LOAD] checkpoint '{name}' 只包含一个平台：{only_platform}。"
                    f"如果你预期 Taobao + Ozon 同时存在，请检查前序输出。"
                )
                self.logger.warning(warning_msg)
                self.memory.add_note("LOAD", warning_msg)

        else:
            self.logger.warning(f"⚠️ [LOAD] {name} checkpoint 未找到 Platform/platform 列。")

        return df

    def _ensure_shared_df_for_step(self, step_id: str) -> None:
        """
        单步续跑时自动加载该步骤所需的最近 checkpoint。
        这样可以支持：
        python -m sop_agent_v4.main --only S5.6
        python -m sop_agent_v4.main --only S6
        """

        if self._shared_df is not None and not self._shared_df.empty:
            return

        step_input_map = {
            "S2": "clean_reviews",
            "S3": "featured_metadata",
            "S3.5": "featured_metadata",
            "S3.6": "semantic_scored",
            "S4": "semantic_scored",
            "S5": "semantic_scored",
            "S5.5": "review_validity_audit",
            "S5.6": "multimodal_base_features",
            "S6": "multimodal_evidence",
        }

        checkpoint_name = step_input_map.get(step_id)

        if not checkpoint_name:
            return

        checkpoint = self._load_checkpoint(checkpoint_name)

        if checkpoint is not None and not checkpoint.empty:
            self._shared_df = checkpoint
            self.logger.info(
                f"✅ 单步续跑自动加载前置 checkpoint: {checkpoint_name} for {step_id}, rows={len(self._shared_df)}"
            )
        else:
            self.logger.warning(
                f"⚠️ 单步续跑需要前置 checkpoint，但未找到: {checkpoint_name}.parquet for {step_id}"
            )

    def _add_artifact_once(self, outputs: Dict[str, str], step_id: str) -> None:
        """
        避免重复注册 artifact。
        """
        for k, v in outputs.items():
            if k not in self.memory.artifacts:
                self.memory.add_artifact(
                    Artifact(
                        name=k,
                        path=str(v),
                        description=f"Output from {step_id}",
                        platform="mixed"
                    )
                )

    def _is_valid_hf_model_dir(self, model_dir: Path) -> bool:
        """
        检查 HuggingFace 模型目录是否完整。
        只存在文件夹不代表模型可用。
        """
        model_dir = Path(model_dir)

        if not model_dir.exists() or not model_dir.is_dir():
            return False

        has_config = (model_dir / "config.json").exists()
        has_weight = (
                (model_dir / "pytorch_model.bin").exists()
                or (model_dir / "model.safetensors").exists()
        )

        has_tokenizer = (
                (model_dir / "tokenizer.json").exists()
                or (model_dir / "vocab.txt").exists()
                or (model_dir / "vocab.json").exists()
        )

        return has_config and has_weight and has_tokenizer

    def execute(self, plan: List[PlanStep]) -> Dict[str, str]:
        outputs: Dict[str, str] = {}
        self.logger.info("🚀 启动全渠道双塔归因流水线 (Thesis Elite Full Cycle)...")
        self._shared_df = None

        for step in plan:
            self.logger.info(f"--- [执行阶段] {step.step_id}: {step.description} ---")

            self._ensure_shared_df_for_step(step.step_id)

            # =========================================================
            # S0: 数据融合
            # =========================================================
            if step.step_id == "S0":
                fusion_results = self.tools.execute_s0_data_fusion()
                outputs.update(fusion_results)

                self.memory.log_step(
                    "S0",
                    "Data Fusion",
                    0,
                    len(fusion_results),
                    "Merged Homepage & Review files",
                    platform="mixed"
                )

            # =========================================================
            # S1: 数据清洗
            # =========================================================
            elif step.step_id == "S1":
                checkpoint = self._load_checkpoint("clean_reviews")
                if checkpoint is not None:
                    self._shared_df = checkpoint

                    # ✅ 关键修复：即使读取旧 featured_metadata，也重新用当前 PLATFORM_WEIGHTS 刷新综合指数
                    if "Platform" in self._shared_df.columns:
                        refreshed = []

                        for platform_name, sub_df in self._shared_df.groupby(
                                self._shared_df["Platform"].astype(str).str.lower()
                        ):
                            sub_df = sub_df.copy()

                            if platform_name in ["taobao", "ozon"]:
                                sub_df = self.tools._add_platform_weighted_index(sub_df, platform_name)

                            refreshed.append(sub_df)

                        self._shared_df = pd.concat(refreshed, ignore_index=True)

                        # 重新保存，避免旧权重继续污染后续步骤
                        refreshed_path = self._save_checkpoint(
                            self._shared_df,
                            "featured_metadata",
                            "S2_refresh_weights"
                        )
                        outputs["featured_data"] = refreshed_path
                    else:
                        outputs["featured_data"] = str(self.tools.paths.output_dir / "featured_metadata.parquet")

                    self.memory.log_step(
                        "S2",
                        "Loaded Existing Feature Engineering and Refreshed OIQ Weights",
                        len(self._shared_df),
                        len(self._shared_df),
                        "Loaded existing featured_metadata checkpoint and recalculated Omnichannel_Capability_Index with current PLATFORM_WEIGHTS",
                        platform="mixed",
                        artifact_name="featured_data"
                    )
                    self._add_artifact_once(outputs, step.step_id)
                    continue

                tb_path = outputs.get("taobao")
                oz_path = outputs.get("ozon")
                dfs = []

                if tb_path:
                    dfs.append(pd.read_excel(tb_path).assign(Platform="taobao"))
                if oz_path:
                    dfs.append(pd.read_excel(oz_path).assign(Platform="ozon"))

                if dfs:
                    raw_df = pd.concat(dfs, ignore_index=True)
                    self._shared_df = self.tools.execute_s1_cleaning(raw_df, "mixed")

                    out_path = self._save_checkpoint(self._shared_df, "clean_reviews", "S1")
                    outputs["clean_reviews"] = out_path

                    self.memory.log_step(
                        "S1", "Cleaning",
                        len(raw_df), len(self._shared_df),
                        "SBERT deduplication applied",
                        platform="mixed",
                        artifact_name="clean_reviews"
                    )

            # =========================================================
            # S1.5: OIQ 权重校准
            # =========================================================
            elif step.step_id == "S1.5":
                weights_df = self.tools.execute_s1_5_oiq_weight_calibration()

                if weights_df is not None and not weights_df.empty:
                    weight_path = self.tools.paths.output_dir / "OIQ_Derived_Platform_Weights.xlsx"
                    outputs["oiq_platform_weights"] = str(weight_path)

                    self.memory.log_step(
                        "S1.5",
                        "OIQ Weight Calibration",
                        len(weights_df),
                        len(weights_df),
                        "Derived platform-specific X1/X2/X3 weights from LLM-based OIQ results",
                        platform="mixed",
                        artifact_name="oiq_platform_weights"
                    )
                else:
                    self.memory.log_step(
                        "S1.5",
                        "OIQ Weight Calibration",
                        0,
                        0,
                        "Skipped or failed; default PLATFORM_WEIGHTS retained",
                        platform="mixed"
                    )

            # =========================================================
            # S2: 特征工程（平台专属）
            # =========================================================
            elif step.step_id == "S2":
                checkpoint = self._load_checkpoint("featured_metadata")
                if checkpoint is not None:
                    self._shared_df = checkpoint
                    outputs["featured_data"] = str(self.tools.paths.output_dir / "featured_metadata.parquet")
                    self.memory.log_step(
                        "S2", "Skipped Existing Feature Engineering",
                        len(self._shared_df), len(self._shared_df),
                        "Loaded existing featured_metadata checkpoint",
                        platform="mixed",
                        artifact_name="featured_data"
                    )
                    self._add_artifact_once(outputs, step.step_id)
                    continue

                if self._shared_df is not None and not self._shared_df.empty:
                    tb_df = self._shared_df[self._shared_df["Platform"].astype(str).str.lower() == "taobao"].copy()
                    oz_df = self._shared_df[self._shared_df["Platform"].astype(str).str.lower() == "ozon"].copy()

                    processed = []

                    if not tb_df.empty:
                        processed.append(self.tools.execute_s2_feature_engineering(tb_df, "taobao"))

                    if not oz_df.empty:
                        processed.append(self.tools.execute_s2_feature_engineering(oz_df, "ozon"))

                    if processed:
                        self._shared_df = pd.concat(processed, ignore_index=True)

                    out_path = self._save_checkpoint(self._shared_df, "featured_metadata", "S2")
                    outputs["featured_data"] = out_path

                    self.memory.log_step(
                        "S2", "Feature Engineering",
                        len(self._shared_df), len(self._shared_df),
                        "Mapped X1/X2/X3 variables",
                        platform="mixed",
                        artifact_name="featured_data"
                    )
            # =========================================================
            # S3: 领域自适应预训练（双塔）
            # =========================================================
            elif step.step_id == "S3":
                data_file = outputs.get("featured_data")

                if not data_file:
                    candidate = self.tools.paths.output_dir / "featured_metadata.parquet"
                    if candidate.exists():
                        data_file = str(candidate)
                        outputs["featured_data"] = str(candidate)
                        self.logger.info(f"✅ S3 自动使用已有 featured_metadata checkpoint: {candidate}")

                cn_model_dir = self.tools.paths.cn_dapt_output or (self.tools.paths.output_dir / "cn_dapt_model")
                ru_model_dir = self.tools.paths.ru_dapt_output or (self.tools.paths.output_dir / "ru_expert_model")

                if data_file:
                    self.logger.info("🔥 S3: 启动双塔 DAPT 训练...")

                    # 中文塔：Taobao
                    if self.tools.paths.cn_base_model:
                        if not self._is_valid_hf_model_dir(Path(cn_model_dir)):
                            self.tools.train_dapt(
                                train_file=data_file,
                                output_path=Path(cn_model_dir),
                                base_model=str(self.tools.paths.cn_base_model),
                                platform="taobao",
                                epochs=3
                            )
                        else:
                            self.logger.info(f"✅ Taobao DAPT 模型已存在且完整，跳过训练: {cn_model_dir}")

                    # 俄语塔：Ozon
                    if self.tools.paths.ru_base_model:
                        if not self._is_valid_hf_model_dir(Path(ru_model_dir)):
                            self.tools.train_dapt(
                                train_file=data_file,
                                output_path=Path(ru_model_dir),
                                base_model=str(self.tools.paths.ru_base_model),
                                platform="ozon",
                                epochs=3
                            )
                        else:
                            self.logger.info(f"✅ Ozon DAPT 模型已存在且完整，跳过训练: {ru_model_dir}")

                outputs["cn_dapt_model"] = str(cn_model_dir)
                outputs["ru_expert_model"] = str(ru_model_dir)

                self.memory.log_step(
                    "S3",
                    "Model Training",
                    0,
                    2,
                    "Dual-Tower DAPT complete/skipped and semantic drift checked",
                    platform="mixed"
                )

            # =========================================================
            # S3.5: 语义推理（生成 BERT_Sentiment_Prob）+ 动态词典
            # =========================================================
            elif step.step_id == "S3.5":
                self.logger.info("🧠 S3.5: 启动专家模型推理 (Real AI Scoring)...")

                checkpoint = self._load_checkpoint("semantic_scored")
                if checkpoint is not None:
                    self._shared_df = checkpoint
                    outputs["semantic_scored"] = str(self.tools.paths.output_dir / "semantic_scored.parquet")
                    self.memory.log_step(
                        "S3.5",
                        "Skipped Existing Semantic Scoring",
                        len(self._shared_df),
                        len(self._shared_df),
                        "Loaded existing semantic_scored checkpoint",
                        platform="mixed",
                        artifact_name="semantic_scored"
                    )
                    self._add_artifact_once(outputs, step.step_id)
                    continue

                if self._shared_df is not None and not self._shared_df.empty:
                    # 先分别做动态词典发现（使用基础模型，不再传 model_path）
                    taobao_df = self._shared_df[
                        self._shared_df["Platform"].astype(str).str.lower() == "taobao"
                        ].copy()

                    ozon_df = self._shared_df[
                        self._shared_df["Platform"].astype(str).str.lower() == "ozon"
                        ].copy()

                    taobao_nebula_path = ""
                    ozon_nebula_path = ""

                    if not taobao_df.empty:
                        self.tools.execute_s2_auto_lexicon_discovery(taobao_df, "taobao")
                        taobao_nebula_path = str(
                            self.tools.paths.output_dir / "Nebula_Lexicon_Data_taobao.xlsx"
                        )
                        outputs["nebula_taobao"] = taobao_nebula_path

                    if not ozon_df.empty:
                        self.tools.execute_s2_auto_lexicon_discovery(ozon_df, "ozon")
                        ozon_nebula_path = str(
                            self.tools.paths.output_dir / "Nebula_Lexicon_Data_ozon.xlsx"
                        )
                        outputs["nebula_ozon"] = ozon_nebula_path
                    # ================================
                    # 🔬 Semantic Drift 检测
                    # 说明：
                    # 1. 必须放在 Nebula 词库生成之后；
                    # 2. 这样 Drift 才能优先使用自动发现的领域词，而不是退回人工词典；
                    # 3. 该结果用于证明 DAPT 是否真正改变了 X1/X2/X3 相关语义空间。
                    # ================================
                    try:
                        cn_model_dir = self.tools.paths.cn_dapt_output or (
                                    self.tools.paths.output_dir / "cn_dapt_model")
                        ru_model_dir = self.tools.paths.ru_dapt_output or (
                                    self.tools.paths.output_dir / "ru_expert_model")

                        if Path(cn_model_dir).exists() and self.tools.paths.cn_base_model:
                            drift_path = self.tools.check_semantic_drift(
                                model_path_new=str(cn_model_dir),
                                model_path_old=str(self.tools.paths.cn_base_model),
                                platform="taobao"
                            )
                            if drift_path:
                                outputs["semantic_drift_taobao"] = drift_path

                        if Path(ru_model_dir).exists() and self.tools.paths.ru_base_model:
                            drift_path = self.tools.check_semantic_drift(
                                model_path_new=str(ru_model_dir),
                                model_path_old=str(self.tools.paths.ru_base_model),
                                platform="ozon"
                            )
                            if drift_path:
                                outputs["semantic_drift_ozon"] = drift_path

                    except Exception as e:
                        self.logger.warning(f"⚠️ Semantic Drift 检测跳过: {e}")

                    # 再注入真实情感概率（使用情感分类模型，不再传 model_path）
                    # 再注入真实情感概率
                    self._shared_df = self.tools.execute_s6_bert_satisfaction_scoring(self._shared_df)

                    out_path = self._save_checkpoint(self._shared_df, "semantic_scored", "S3.5")
                    outputs["semantic_scored"] = out_path

                    if "BERT_Sentiment_Prob" in self._shared_df.columns:
                        self.logger.info("✅ 真实语义概率已注入 (Y变量地基)")
                    else:
                        self.logger.error("❌ 语义打分失败，回归模型将失去因变量！")

                    self.memory.log_step(
                        "S3.5",
                        "Semantic Inference",
                        len(self._shared_df),
                        len(self._shared_df),
                        "Generated BERT_Sentiment_Prob and nebula lexicon",
                        platform="mixed",
                        artifact_name="semantic_scored"
                    )
                if "BERT_Sentiment_Prob" not in self._shared_df.columns:
                    raise RuntimeError("S3.5 failed: BERT_Sentiment_Prob not generated.")

            # =========================================================
            # S3.6: BERT-based Review Validity Audit
            # =========================================================
            elif step.step_id == "S3.6":
                self.logger.info("🔎 S3.6: 启动基于 BERT 的评论有效性审计...")

                checkpoint = self._load_checkpoint("review_validity_audit")
                if checkpoint is not None:
                    self._shared_df = checkpoint
                    outputs["review_validity_audit"] = str(
                        self.tools.paths.output_dir / "review_validity_audit.parquet"
                    )
                    self.memory.log_step(
                        "S3.6",
                        "Skipped Existing Review Validity Audit",
                        len(self._shared_df),
                        len(self._shared_df),
                        "Loaded existing review_validity_audit checkpoint",
                        platform="mixed",
                        artifact_name="review_validity_audit"
                    )
                    self._add_artifact_once(outputs, step.step_id)
                    continue

                if self._shared_df is not None and not self._shared_df.empty:
                    before_rows = len(self._shared_df)

                    self._shared_df = self.tools.execute_s3_6_review_validity_audit(self._shared_df)

                    out_path = self._save_checkpoint(
                        self._shared_df,
                        "review_validity_audit",
                        "S3.6"
                    )

                    outputs["review_validity_audit"] = out_path
                    outputs["review_validity_summary"] = str(
                        self.tools.paths.output_dir / "Review_Validity_Audit_Summary.xlsx"
                    )

                    invalid_count = (
                        int(self._shared_df["Is_Invalid_Review"].sum())
                        if "Is_Invalid_Review" in self._shared_df.columns
                        else 0
                    )

                    self.memory.log_step(
                        "S3.6",
                        "BERT-based Review Validity Audit",
                        before_rows,
                        len(self._shared_df),
                        f"Generated Invalid_Risk_Score and Review_Validity_Weight; invalid_count={invalid_count}",
                        platform="mixed",
                        artifact_name="review_validity_audit"
                    )
                if "Review_Validity_Weight" not in self._shared_df.columns:
                    raise RuntimeError("S3.6 failed: Review_Validity_Weight not generated.")

            # =========================================================
            # S4: 全量十折交叉验证 PPL 审计
            # =========================================================
            elif step.step_id == "S4":
                self.logger.info("🧪 S4: 启动全量十折交叉验证 PPL 审计...")

                data_file = outputs.get("semantic_scored") or outputs.get("featured_data")

                if not data_file:
                    candidate = self.tools.paths.output_dir / "semantic_scored.parquet"
                    if candidate.exists():
                        data_file = str(candidate)
                        outputs["semantic_scored"] = str(candidate)
                        self.logger.info(f"✅ S4 自动使用已有 semantic_scored checkpoint: {candidate}")

                if not data_file:
                    candidate = self.tools.paths.output_dir / "featured_metadata.parquet"
                    if candidate.exists():
                        data_file = str(candidate)
                        outputs["featured_data"] = str(candidate)
                        self.logger.info(f"✅ S4 自动使用已有 featured_metadata checkpoint: {candidate}")

                if not data_file:
                    self.logger.warning("⚠️ S4 缺少输入数据 semantic_scored / featured_data，跳过。")
                    self.memory.log_step(
                        "S4",
                        "Skipped CV Validation",
                        0,
                        0,
                        "Missing semantic_scored or featured_data",
                        platform="mixed"
                    )
                    continue

                cn_model_dir = self.tools.paths.cn_dapt_output or (self.tools.paths.output_dir / "cn_dapt_model")
                ru_model_dir = self.tools.paths.ru_dapt_output or (self.tools.paths.output_dir / "ru_expert_model")

                cv_outputs = {}

                if Path(cn_model_dir).exists():
                    cv_taobao = self.tools.run_cv_validation(
                        model_path=str(cn_model_dir),
                        data_file=data_file,
                        platform="taobao"
                    )
                    if cv_taobao:
                        outputs["cv_report_taobao"] = cv_taobao
                        cv_outputs["taobao"] = cv_taobao
                else:
                    self.logger.warning(f"⚠️ 中文 DAPT 模型不存在，跳过 Taobao CV: {cn_model_dir}")

                if Path(ru_model_dir).exists():
                    cv_ozon = self.tools.run_cv_validation(
                        model_path=str(ru_model_dir),
                        data_file=data_file,
                        platform="ozon"
                    )
                    if cv_ozon:
                        outputs["cv_report_ozon"] = cv_ozon
                        cv_outputs["ozon"] = cv_ozon
                else:
                    self.logger.warning(f"⚠️ 俄语 DAPT 模型不存在，跳过 Ozon CV: {ru_model_dir}")

                self.memory.register_step_output("S4", {
                    "cv_outputs": cv_outputs,
                    "input_data": data_file,
                })

                self.memory.log_step(
                    "S4",
                    "Full-data 10-fold CV",
                    0,
                    len(cv_outputs),
                    "Generated platform-specific PPL reports",
                    platform="mixed",
                    artifact_name="cv_report"
                )

            # =========================================================
            # S5: 权重健康检查 + 语义完整性审计
            # =========================================================
            elif step.step_id == "S5":
                self.logger.info("🔬 S5: 启动模型权重健康与语义完整性审计...")

                cn_model_dir = self.tools.paths.cn_dapt_output or (self.tools.paths.output_dir / "cn_dapt_model")
                ru_model_dir = self.tools.paths.ru_dapt_output or (self.tools.paths.output_dir / "ru_expert_model")

                audit_outputs = {}

                if Path(cn_model_dir).exists():
                    tb_audit = self.tools.run_s5_model_audit(
                        model_path=str(cn_model_dir),
                        platform="taobao"
                    )
                    outputs.update(tb_audit)
                    audit_outputs.update(tb_audit)
                else:
                    self.logger.warning(f"⚠️ 中文 DAPT 模型不存在，跳过 Taobao S5 审计: {cn_model_dir}")

                if Path(ru_model_dir).exists():
                    oz_audit = self.tools.run_s5_model_audit(
                        model_path=str(ru_model_dir),
                        platform="ozon"
                    )
                    outputs.update(oz_audit)
                    audit_outputs.update(oz_audit)
                else:
                    self.logger.warning(f"⚠️ 俄语 DAPT 模型不存在，跳过 Ozon S5 审计: {ru_model_dir}")

                self.memory.register_step_output("S5", {
                    "audit_outputs": audit_outputs,
                    "cn_model_dir": str(cn_model_dir),
                    "ru_model_dir": str(ru_model_dir),
                })

                self.memory.log_step(
                    "S5",
                    "Model Sanity and Semantic Audit",
                    0,
                    len(audit_outputs),
                    "Generated sanity reports and semantic audit files",
                    platform="mixed",
                    artifact_name="model_audit"
                )

            # =========================================================
            # S5.5: 多模态基础字段
            # =========================================================
            elif step.step_id == "S5.5":

                checkpoint = self._load_checkpoint("multimodal_base_features")
                if checkpoint is not None:
                    self._shared_df = checkpoint
                    outputs["multimodal_base_features"] = str(
                        self.tools.paths.output_dir / "multimodal_base_features.parquet")
                    self.memory.log_step(
                        "S5.5",
                        "Skipped Existing Multimodal Base Features",
                        len(self._shared_df),
                        len(self._shared_df),
                        "Loaded existing multimodal_base_features checkpoint",
                        platform="mixed",
                        artifact_name="multimodal_base_features"
                    )
                    self._add_artifact_once(outputs, step.step_id)
                    continue

                if self._shared_df is not None and not self._shared_df.empty:
                    self._shared_df = self.tools.execute_s5_5_multimodal_base_features(self._shared_df)

                    out_path = self._save_checkpoint(self._shared_df, "multimodal_base_features", "S5.5")
                    outputs["multimodal_base_features"] = out_path

                    self.memory.log_step(
                        "S5.5",
                        "Multimodal Base Features",
                        len(self._shared_df),
                        len(self._shared_df),
                        "Generated seller/review media presence and count fields",
                        platform="mixed",
                        artifact_name="multimodal_base_features"
                    )
                if self._shared_df is None or self._shared_df.empty:
                    raise RuntimeError(
                        "S5.5 failed: missing input dataframe. "
                        "Please run S3.5 first to generate semantic_scored.parquet."
                    )

            # =========================================================
            # S5.6: 多模态证据变量（CLIP版）
            # =========================================================
            elif step.step_id == "S5.6":
                checkpoint = self._load_checkpoint("multimodal_evidence")
                if checkpoint is not None:
                    self._shared_df = checkpoint
                    outputs["multimodal_evidence"] = str(
                        self.tools.paths.output_dir / "multimodal_evidence.parquet"
                    )
                    self.memory.log_step(
                        "S5.6",
                        "Skipped Existing Multimodal Evidence",
                        len(self._shared_df),
                        len(self._shared_df),
                        "Loaded existing multimodal_evidence checkpoint",
                        platform="mixed",
                        artifact_name="multimodal_evidence"
                    )
                    self._add_artifact_once(outputs, step.step_id)
                    continue

                if self._shared_df is not None and not self._shared_df.empty:
                    self._shared_df = self.tools.execute_s5_6_multimodal_evidence(self._shared_df)

                    out_path = self._save_checkpoint(self._shared_df, "multimodal_evidence", "S5.6")
                    outputs["multimodal_evidence"] = out_path

                    self.memory.log_step(
                        "S5.6",
                        "Multimodal Evidence",
                        len(self._shared_df),
                        len(self._shared_df),
                        "Generated multimodal evidence MVP fields",
                        platform="mixed",
                        artifact_name="multimodal_evidence"
                    )
                if self._shared_df is None or self._shared_df.empty:
                    raise RuntimeError(
                        "S5.6 failed: missing input dataframe. "
                        "Please run S5.5 first to generate multimodal_base_features.parquet."
                    )

                if "Evidence_review" not in self._shared_df.columns:
                    raise RuntimeError("S5.6 failed: Evidence_review not generated.")

            # =========================================================
            # S6: CSI + 回归 + 可视化
            # =========================================================
            elif step.step_id == "S6":
                if self._shared_df is not None and not self._shared_df.empty:
                    # 1. 计算 CSI
                    self._shared_df = self.tools.execute_s6_csi_scoring(self._shared_df)

                    # 2. 导出最终回归输入表
                    out_path = self._save_checkpoint(self._shared_df, "S6_Final_Regression_Input", "S6")
                    outputs["final_regression_input"] = out_path

                    # 3. 执行回归
                    outputs["regression_report"] = self.tools.run_scientific_regression(out_path)

                    # 4. 雷达图
                    outputs["radar_chart"] = self.tools.plot_radar_chart_elite(out_path)

                    self.memory.log_step(
                        "S6",
                        "Final Attribution",
                        len(self._shared_df),
                        1,
                        "HC3 regression and radar chart generated",
                        platform="mixed",
                        artifact_name="regression_report"
                    )

            # =========================================================
            # 每步归档
            # =========================================================
            self._add_artifact_once(outputs, step.step_id)

        return outputs