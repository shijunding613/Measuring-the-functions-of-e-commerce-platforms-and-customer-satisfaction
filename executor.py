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

    def execute(self, plan: List[PlanStep]) -> Dict[str, str]:
        outputs: Dict[str, str] = {}
        self.logger.info("🚀 启动全渠道双塔归因流水线 (Thesis Elite Full Cycle)...")
        self._shared_df = None

        for step in plan:
            self.logger.info(f"--- [执行阶段] {step.step_id}: {step.description} ---")

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

                    out_path = self.tools.paths.output_dir / "clean_reviews.parquet"
                    self._shared_df.to_parquet(out_path, index=False)
                    outputs["clean_reviews"] = str(out_path)

                    self.memory.log_step(
                        "S1",
                        "Cleaning",
                        len(raw_df),
                        len(self._shared_df),
                        "SBERT deduplication applied",
                        platform="mixed",
                        artifact_name="clean_reviews"
                    )

            # =========================================================
            # S2: 特征工程（平台专属）
            # =========================================================
            elif step.step_id == "S2":
                if self._shared_df is not None and not self._shared_df.empty:
                    tb_df = self._shared_df[self._shared_df["Platform"].astype(str).str.lower() == "taobao"].copy()
                    oz_df = self._shared_df[self._shared_df["Platform"].astype(str).str.lower() == "ozon"].copy()

                    processed = []

                    if not tb_df.empty:
                        tb_feat = self.tools.execute_s2_feature_engineering(tb_df, "taobao")
                        processed.append(tb_feat)

                    if not oz_df.empty:
                        oz_feat = self.tools.execute_s2_feature_engineering(oz_df, "ozon")
                        processed.append(oz_feat)

                    if processed:
                        self._shared_df = pd.concat(processed, ignore_index=True)

                    feat_path = self.tools.paths.output_dir / "featured_metadata.parquet"
                    self._shared_df.to_parquet(feat_path, index=False)
                    outputs["featured_data"] = str(feat_path)

                    self.memory.log_step(
                        "S2",
                        "Feature Engineering",
                        len(self._shared_df),
                        len(self._shared_df),
                        "Mapped X1/X2/X3 variables",
                        platform="mixed",
                        artifact_name="featured_data"
                    )

            # =========================================================
            # S3: 领域自适应预训练（双塔）
            # =========================================================
            elif step.step_id == "S3":
                data_file = outputs.get("featured_data")

                cn_model_dir = self.tools.paths.cn_dapt_output or (self.tools.paths.output_dir / "cn_dapt_model")
                ru_model_dir = self.tools.paths.ru_dapt_output or (self.tools.paths.output_dir / "ru_expert_model")

                if data_file:
                    self.logger.info("🔥 S3: 启动双塔 DAPT 训练...")

                    # 中文塔：Taobao
                    if self.tools.paths.cn_base_model:
                        if not Path(cn_model_dir).exists():
                            self.tools.train_dapt(
                                train_file=data_file,
                                output_path=Path(cn_model_dir),
                                base_model=str(self.tools.paths.cn_base_model),
                                platform="taobao",
                                epochs=3
                            )

                    # 俄语塔：Ozon
                    if self.tools.paths.ru_base_model:
                        if not Path(ru_model_dir).exists():
                            self.tools.train_dapt(
                                train_file=data_file,
                                output_path=Path(ru_model_dir),
                                base_model=str(self.tools.paths.ru_base_model),
                                platform="ozon",
                                epochs=3
                            )

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
                    self._shared_df = self.tools.execute_s6_bert_satisfaction_scoring(self._shared_df)

                    semantic_path = self.tools.paths.output_dir / "semantic_scored.parquet"
                    self._shared_df.to_parquet(semantic_path, index=False)
                    outputs["semantic_scored"] = str(semantic_path)

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

            # =========================================================
            # S4 / S5: 当前版本未启用
            # =========================================================
            elif step.step_id == "S4":
                self.logger.info("S4: 跳过（当前版本未启用）")

            elif step.step_id == "S5":
                self.logger.info("S5: 跳过（当前版本未启用）")

            # =========================================================
            # S5.5: 多模态基础字段
            # =========================================================
            elif step.step_id == "S5.5":
                if self._shared_df is not None and not self._shared_df.empty:
                    self._shared_df = self.tools.execute_s5_5_multimodal_base_features(self._shared_df)

                    multimodal_base_path = self.tools.paths.output_dir / "multimodal_base_features.parquet"
                    self._shared_df.to_parquet(multimodal_base_path, index=False)
                    outputs["multimodal_base_features"] = str(multimodal_base_path)

                    self.memory.log_step(
                        "S5.5",
                        "Multimodal Base Features",
                        len(self._shared_df),
                        len(self._shared_df),
                        "Generated seller/review media presence and count fields",
                        platform="mixed",
                        artifact_name="multimodal_base_features"
                    )

            # =========================================================
            # S5.6: 多模态证据变量（CLIP版）
            # =========================================================
            elif step.step_id == "S5.6":
                if self._shared_df is not None and not self._shared_df.empty:
                    self._shared_df = self.tools.execute_s5_6_multimodal_evidence(self._shared_df)

                    multimodal_evidence_path = self.tools.paths.output_dir / "multimodal_evidence.parquet"
                    self._shared_df.to_parquet(multimodal_evidence_path, index=False)
                    outputs["multimodal_evidence"] = str(multimodal_evidence_path)

                    self.memory.log_step(
                        "S5.6",
                        "Multimodal Evidence",
                        len(self._shared_df),
                        len(self._shared_df),
                        "Generated multimodal evidence MVP fields",
                        platform="mixed",
                        artifact_name="multimodal_evidence"
                    )

            # =========================================================
            # S6: CSI + 回归 + 可视化
            # =========================================================
            elif step.step_id == "S6":
                if self._shared_df is not None and not self._shared_df.empty:
                    # 1. 计算 CSI
                    self._shared_df = self.tools.execute_s6_csi_scoring(self._shared_df)

                    # 2. 导出最终回归输入表
                    final_path = self.tools.paths.output_dir / "S6_Final_Regression_Input.parquet"
                    self._shared_df.to_parquet(final_path, index=False)
                    outputs["final_regression_input"] = str(final_path)

                    # 3. 执行回归
                    outputs["regression_report"] = self.tools.run_scientific_regression(str(final_path))

                    # 4. 雷达图
                    outputs["radar_chart"] = self.tools.plot_radar_chart_elite(str(final_path))

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
            for k, v in outputs.items():
                if k not in self.memory.artifacts:
                    self.memory.add_artifact(
                        Artifact(
                            name=k,
                            path=str(v),
                            description=f"Output from {step.step_id}",
                            platform="mixed"
                        )
                    )

        return outputs