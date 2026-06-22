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
                inputs=["taobao_raw_dir", "ozon_raw_dir", "taobao_homepage", "ozon_homepage"],
                outputs=["Master_Data_Taobao.xlsx", "Master_Data_Ozon.xlsx"],
            ),

            PlanStep(
                step_id="S1",
                title="宽表语义降维清洗",
                description=(
                    "合并 Taobao 与 Ozon 宽表，统一 Platform 字段。"
                    "执行基础去重、HTML 清洗、文本长度标记，并按平台分别对 Taobao 与 Ozon 启用 SBERT 语义去重。"
                ),
                inputs=["Master_Data_Taobao.xlsx", "Master_Data_Ozon.xlsx"],
                outputs=["clean_reviews.parquet", "clean_reviews.xlsx"],
            ),

            PlanStep(
                step_id="S1.5",
                title="OIQ-derived Platform Weight Calibration",
                description=(
                    "读取 2.3 节 LLM/OIQ 评论识别结果，"
                    "根据 OIQ 维度出现率反推 Taobao 与 Ozon 的 X1/X2/X3 平台权重，"
                    "并输出 OIQ_Derived_Platform_Weights.xlsx，作为 2.4 节综合指数权重来源。"
                ),
                inputs=["LLM_OIQ_评论识别结果.xlsx"],
                outputs=["OIQ_Derived_Platform_Weights.xlsx"],
            ),

            PlanStep(
                step_id="S2",
                title="双塔物理特征权重映射 (X1, X2, X3)",
                description=(
                    "1. 淘宝 (S-D Logic): 量化富媒体交互、价格剪刀差、人本服务响应与生态锁定。\n"
                    "2. Ozon (System Logic): 量化 FBO 基建红利、制度性退货、社区互动与金融绿卡粘性。"
                ),
                inputs=["clean_reviews.parquet"],
                outputs=["featured_metadata.parquet", "featured_metadata.xlsx"],
            ),

            PlanStep(
                step_id="S3",
                title="领域自适应预训练 (Elite DAPT)",
                description=(
                    "采用中文 RoBERTa-WWM 与俄语 RuBERT 基础模型进行领域自适应预训练。"
                    "如本地 DAPT 模型目录已存在，则自动跳过重复训练。"
                ),
                inputs=["featured_metadata.parquet", "cn_base_model", "ru_base_model"],
                outputs=["cn_dapt_model", "ru_expert_model", "Model_Evolution_Manifest_*.xlsx"],
            ),

            PlanStep(
                step_id="S3.5",
                title="语义打分、星云词库与语义漂移检测",
                description=(
                    "1. 生成 BERT_Sentiment_Prob，作为 CSI 因变量的语义基础。\n"
                    "2. 分平台生成 Nebula_Lexicon_Data_taobao / ozon 词库。\n"
                    "3. 对 DAPT 前后模型进行 Semantic Drift 检测，用于解释领域知识注入效果。"
                ),
                inputs=["featured_metadata.parquet", "cn_dapt_model", "ru_expert_model"],
                outputs=[
                    "semantic_scored.parquet",
                    "semantic_scored.xlsx",
                    "Nebula_Lexicon_Data_taobao.xlsx",
                    "Nebula_Lexicon_Data_ozon.xlsx",
                    "Semantic_Drift_taobao.xlsx",
                    "Semantic_Drift_ozon.xlsx",
                ],
            ),

            PlanStep(
                step_id="S3.6",
                title="BERT-based Review Validity Audit",
                description=(
                    "基于已生成的 BERT_Sentiment_Prob、文本长度、语义维度信号与重复文本特征，"
                    "识别低信息量评论、极端短文本评论、弱分析价值评论和疑似异常评论。"
                    "该步骤不判断法律意义上的虚假评论，而是用于 Ozon 无效评论问题的稳健性控制。"
                ),
                inputs=["semantic_scored.parquet"],
                outputs=["review_validity_audit.parquet", "Review_Validity_Audit_Summary.xlsx"],
            ),

            PlanStep(
                step_id="S4",
                title="全量十折交叉验证 (PPL 审计)",
                description=(
                    "分别对 Taobao 中文 DAPT 模型与 Ozon 俄语 DAPT 模型执行平台专属 10 折交叉验证。"
                    "输出 cv_report_taobao.json 与 cv_report_ozon.json，用于验证领域自适应模型在电商评论语料上的泛化能力。"
                ),
                inputs=["semantic_scored.parquet", "cn_dapt_model", "ru_expert_model"],
                outputs=["cv_report_taobao.json", "cv_report_ozon.json"],
            ),

            PlanStep(
                step_id="S5",
                title="权重健康检查与语义完整性审计",
                description=(
                    "分别对 Taobao 与 Ozon 的 DAPT 模型执行权重健康检查、LayerNorm 诊断与 SVD 语义空间完整性审计。"
                    "输出 sanity_report、Semantic_Audit 与 Executive_Summary，用于论文中证明模型未发生权重坍缩。"
                ),
                inputs=["cn_dapt_model", "ru_expert_model"],
                outputs=[
                    "sanity_report_taobao.json",
                    "sanity_report_ozon.json",
                    "Semantic_Audit_taobao.xlsx",
                    "Semantic_Audit_ozon.xlsx",
                    "Executive_Summary_taobao.txt",
                    "Executive_Summary_ozon.txt",
                ],
            ),

            PlanStep(
                step_id="S5.5",
                title="多模态基础字段生成",
                description=(
                    "读取本地主页图片、评论图片与视频字段，生成 Seller_Image_Count、"
                    "Review_First_Image_Path、Has_Image、Has_Video、Media_Count 等基础多模态变量。"
                ),
                inputs=["semantic_scored.parquet", "local_image_folders"],
                outputs=["multimodal_base_features.parquet", "multimodal_base_features.xlsx"],
            ),

            PlanStep(
                step_id="S5.6",
                title="CLIP 多模态证据变量构建",
                description=(
                    "强制执行 CLIP Text-Image Alignment，生成 Text_Image_Alignment、"
                    "Evidence_image、Evidence_video、CredibleAlign 与 Evidence_review。"
                    "本节点不读取旧 multimodal_evidence checkpoint，便于首次正式运行 CLIP。"
                ),
                inputs=["multimodal_base_features.parquet", "clip_model", "Review_First_Image_Path"],
                outputs=["multimodal_evidence.parquet", "multimodal_evidence.xlsx"],
            ),

            PlanStep(
                step_id="S6",
                title="终极归因分析与科学可视化",
                description=(
                    "1. 计算 CSI: Y = Map5(BERT_Sentiment_Prob + Social_Signals)。\n"
                    "2. 淘宝引入 Helpful 放大器，Ozon 引入 log((Yes+1)/(No+1)) 社区共识修正。\n"
                    "3. 执行 HC3 稳健回归与 VIF 审计，产出最终回归报告、雷达图与描述性统计。"
                ),
                inputs=["multimodal_evidence.parquet"],
                outputs=[
                    "S6_Final_Regression_Input.parquet",
                    "S6_Final_Regression_Input.xlsx",
                    "Regression_Final_Report.txt",
                    "Omnichannel_Radar_Elite.png",
                    "descriptive_stats.xlsx",
                    "Master_Data_FINAL_THESIS.xlsx",
                ],
            ),
        ]