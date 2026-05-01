from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd


# --- 环境初始化 ---
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

try:
    from sop_agent_v4.planner import SOPPlanner
    from sop_agent_v4.executor import SOPExecutor
    from sop_agent_v4.memory import AgentMemory
    from sop_agent_v4.tools import LocalPaths, LocalTools
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def generate_descriptive_stats(df: pd.DataFrame, output_dir: Path) -> None:
    """
    生成描述性统计表
    """
    if df is None or df.empty:
        return

    target_vars = [
        "CSI_Score",
        "X1_Logistics",
        "X2_Service",
        "X3_Eco",
        "Evidence_review",
        "Control_Price",
        "Control_Quality",
        "Control_Pop",
        "Control_Scarcity",
    ]

    existing_vars = [c for c in target_vars if c in df.columns]
    if not existing_vars:
        return

    platform_col = "Platform" if "Platform" in df.columns else ("platform" if "platform" in df.columns else None)

    try:
        if platform_col:
            stats = df.groupby(platform_col)[existing_vars].describe().T
        else:
            stats = df[existing_vars].describe().T

        stats_path = output_dir / "descriptive_stats.xlsx"
        stats.to_excel(stats_path)
        logging.info(f"📊 描述性统计已输出: {stats_path}")
    except Exception as e:
        logging.warning(f"⚠️ 描述性统计输出失败: {e}")


def save_reports(output_dir: Path, reports: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "cv_report.json"

    def make_json_safe(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_json_safe(x) for x in obj]
        return obj

    safe_reports = make_json_safe(reports)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(safe_reports, f, ensure_ascii=False, indent=2)
    logging.info(f"📝 JSON 报告已保存: {report_path}")


def build_config() -> LocalPaths:
    """
    构建本地路径配置
    """
    project_root = Path(
        r"D:\论文\硕士论文\量化数据资料\模型结果\模型环境\sop_agent_project_3_Multi_source_Heterogeneous_Data_Fusion"
    )

    output_dir = project_root / "outputs"

    cfg = LocalPaths(
        taobao_raw_dir=project_root / "淘宝评论初步清洗汇总",
        ozon_raw_dir=project_root / "ozon评论初步清洗汇总",
        output_dir=output_dir,
        taobao_homepage=project_root / "Taobao_Homepage_Cleaned_Final_media.xlsx",
        ozon_homepage=project_root / "Ozon_Homepage_Cleaned_Final_media.xlsx",

        # -----------------------------
        # 基础模型（给 S2.5 词典发现 / S3 DAPT 起点）
        # -----------------------------
        cn_base_model=r"C:\Users\June\.cache\huggingface\hub\models--hfl--chinese-roberta-wwm-ext\snapshots\5c58d0b8ec1d9014354d691c538661bf00bfdb44",
        ru_base_model=r"C:\Users\June\.cache\huggingface\hub\models--DeepPavlov--rubert-base-cased\snapshots\4036cab694767a299f2b9e6492909664d9414229",

        # -----------------------------
        # DAPT 输出目录（只作为训练产物）
        # -----------------------------
        cn_dapt_output=output_dir / "cn_dapt_model",
        ru_dapt_output=output_dir / "ru_expert_model",

        # -----------------------------
        # 情感分类模型（给 S3.5 / S6 真实情感概率）
        # -----------------------------
        cn_sentiment_model="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
        ru_sentiment_model="cointegrated/rubert-tiny-sentiment-balanced",
        clip_model=r"C:\Users\June\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\你的实际snapshot",

        taobao_home_img_dir=project_root / "淘宝主页产品图片",
        ozon_home_img_dir=project_root / "ozon主页产品图片",
        taobao_review_img_dir=project_root / "淘宝评论初步清洗汇总" / "淘宝评价图片",
        ozon_review_img_dir=project_root / "ozon评论初步清洗汇总" / "ozon评价图片",
    )

    return cfg


def main() -> None:
    cfg = build_config()
    setup_logging(cfg.output_dir)

    logger = logging.getLogger("main")
    logger.info("🚀 启动 SOP Agent v4 多源异构数据融合流程")

    memory = AgentMemory()
    tools = LocalTools(cfg)
    planner = SOPPlanner()
    executor = SOPExecutor(tools=tools, memory=memory)
    plan = planner.build_plan()
    outputs = executor.execute(plan)

    # ================================
    # 📊 导出研究审计报告
    # ================================
    memory.export_audit_report(cfg.output_dir)
    memory.save_research_manifest(str(cfg.output_dir / "research_manifest.json"))

    try:
        plan = planner.build_plan()
        reports = executor.execute(plan)

        final_table_path = cfg.output_dir / "S6_Final_Regression_Input.parquet"
        if final_table_path.exists():
            final_df = pd.read_parquet(final_table_path)

            output_table = cfg.output_dir / "Master_Data_FINAL_THESIS.xlsx"
            final_df.to_excel(output_table, index=False)
            logger.info(f"✅ 最终总表已保存: {output_table}")

            generate_descriptive_stats(final_df, cfg.output_dir)
        else:
            logger.warning("⚠️ 未找到最终总表 parquet，跳过 Excel 与描述性统计输出。")

        if reports:
            save_reports(cfg.output_dir, reports)

        logger.info("🎉 流程执行完成")

    except Exception as e:
        logger.exception(f"❌ 主流程执行失败: {e}")
        raise


if __name__ == "__main__":
    main()