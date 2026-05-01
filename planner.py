"""
Planner for the e-commerce review mining SOP (Thesis Elite Version).
严格对齐《回归模型权重设计》：
1. 区分 Taobao S-D Logic 与 Ozon System Logic。
2. 引入 CSI 复合满意度算法。
3. 强化 VIF + HC3 稳健回归审计。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class PlanStep:
    step_id: str
    title: str
    description: str
    inputs: List[str]
    outputs: List[str]

class SOPPlanner:
    """构建与《权重设计方案》严格对齐的执行计划。"""

    def build_plan(self) -> List[PlanStep]:
        return [
            PlanStep(
                step_id="S0",
                title="全字段归因映射融合 (Data Fusion)",
                description=(
                    "执行主页数据与评论数据的全字段映射。"
                    "淘宝侧集成发货时长、客服响应、视频交互等元数据。"
                    "Ozon 侧集成 FBO 仓储状态、绿卡价格、退货政策等系统指标。"
                ),
                inputs=["taobao_raw_dir", "ozon_raw_dir", "homepage_context"],
                outputs=["Master_Data_Taobao.xlsx", "Master_Data_Ozon.xlsx"],
            ),
            PlanStep(
                step_id="S1",
                title="宽表语义降维清洗",
                description=(
                    "执行 SBERT 语义去重，利用 NLP 过滤 HTML 噪音。"
                    "针对 Ozon 俄语评论进行特殊字符处理与无效差评初步识别。"
                ),
                inputs=["Master_Data_*.xlsx"],
                outputs=["clean_reviews.parquet"],
            ),
            PlanStep(
                step_id="S2",
                title="双塔物理特征权重映射 (X1, X2, X3)",
                description=(
                    "1. 淘宝 (S-D Logic): 量化富媒体交互($X_2$)、价格剪刀差($X_3$)与人本服务响应。\n"
                    "2. Ozon (System Logic): 量化 FBO 基建红利($X_1$)、制度性退货($X_2$)与金融绿卡粘性($X_3$)。"
                ),
                inputs=["clean_reviews.parquet"],
                outputs=["featured_metadata.parquet"],
            ),
            PlanStep(
                step_id="S3",
                title="领域自适应预训练 (Elite DAPT)",
                description=(
                    "采用 RoBERTa-WWM (CN) 与 LLRD (RU) 模型重塑电商语义表征。"
                    "针对“必赔”、“极速达”等高频垂直领域词汇进行知识注入。"
                ),
                inputs=["featured_metadata.parquet"],
                outputs=["cn_dapt_model", "ru_expert_model"],
            ),
            PlanStep(
                step_id="S3.5",
                title="语义打分与星云词库挖掘",
                description=(
                    "1. 利用专家模型推理真实情感概率 (BERT Probability)，构建因变量 $Y$ 的语义地基。\n"
                    "2. 动态扫描并生成行业黑话星云图 (Nebula Lexicon)。"
                ),
                inputs=["cn_dapt_model", "featured_metadata.parquet"],
                outputs=["semantic_scored_data.parquet", "Topic_Keywords_*.xlsx"],
            ),
            PlanStep(
                step_id="S4",
                title="全量十折交叉验证 (PPL 审计)",
                description="执行 MLM 任务的 Loss 审计，计算模型在电商文本上的困惑度 (PPL)，验证泛化能力。",
                inputs=["models", "semantic_scored_data.parquet"],
                outputs=["cv_report.json"],
            ),
            PlanStep(
                step_id="S5",
                title="权重偏移与语义完整性审计",
                description="通过 SVD 奇异值分解检查模型是否存在维度坍缩，产出用于论文的《执行摘要报告》。",
                inputs=["models"],
                outputs=["Weight_Diagnosis_*.xlsx", "Executive_Summary_*.txt"],
            ),
            PlanStep(
                step_id="S5.5",
                title="多模态基础字段生成",
                description=(
                    "生成卖家媒体供给与评论媒体证据的基础字段，包括 Product_Image/Product_Video、"
                    "Review_Images/Review_Videos 的存在性与数量变量。"
                ),
                inputs=["semantic_scored_data.parquet"],
                outputs=["multimodal_base_features.parquet"],
            ),
            PlanStep(
                step_id="S5.6",
                title="多模态证据变量构建",
                description=(
                    "构建 Text-Image / Text-Video 对齐、Noise Gating、CredibleAlign 与 Evidence_review 等多模态证据变量。"
                ),
                inputs=["multimodal_base_features.parquet"],
                outputs=["multimodal_evidence.parquet"],
            ),
            PlanStep(
                step_id="S6",
                title="终极归因分析与科学可视化",
                description=(
                    "1. 计算 CSI: $Y = \text{Map5}(\text{BERT} + \text{Social\_Signals})$。\n"
                    "2. 淘宝引入 Helpful 放大器，Ozon 引入 log((Yes+1)/(No+1)) 形式的社区共识修正。\n"
                    "3. 执行 HC3 稳健回归与 VIF 审计，产出效用雷达图。"
                ),
                inputs=["semantic_scored_data.parquet", "multimodal_evidence.parquet"],
                outputs=["Regression_Final_Report.txt", "Omnichannel_Radar_Elite.png", "Thesis_Descriptive_Statistics.xlsx"],
            ),
        ]