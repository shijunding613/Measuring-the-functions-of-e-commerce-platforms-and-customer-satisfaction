from __future__ import annotations
import argparse
import json
import logging
import os
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
    生成论文可用的描述性统计表：
    Platform / Variable / N / Mean / Std / Min / Median / Max
    """
    if df is None or df.empty:
        return

    target_vars = [
        "CSI_Score",
        "BERT_Sentiment_Prob",
        "Invalid_Risk_Score",
        "Review_Validity_Weight",
        "Text_Image_Alignment",
        "Evidence_image",
        "Evidence_video",
        "Evidence_review",
        "CredibleAlign",
        "X1_Logistics",
        "X2_Service",
        "X3_Eco",
        "X1_Logistics_Norm",
        "X2_Service_Norm",
        "X3_Eco_Norm",
        "Omnichannel_Capability_Index",
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
        rows = []

        if platform_col:
            groups = df.groupby(df[platform_col].astype(str).str.lower())
        else:
            groups = [("mixed", df)]

        for platform, sub in groups:
            for var in existing_vars:
                s = pd.to_numeric(sub[var], errors="coerce").dropna()

                rows.append({
                    "Platform": platform,
                    "Variable": var,
                    "N": int(s.count()),
                    "Mean": round(float(s.mean()), 6) if len(s) else None,
                    "Std": round(float(s.std()), 6) if len(s) > 1 else None,
                    "Min": round(float(s.min()), 6) if len(s) else None,
                    "Median": round(float(s.median()), 6) if len(s) else None,
                    "Max": round(float(s.max()), 6) if len(s) else None,
                })

        stats_df = pd.DataFrame(rows)

        stats_path = output_dir / "descriptive_stats.xlsx"
        stats_df.to_excel(stats_path, index=False)

        logging.info(f"📊 描述性统计已输出: {stats_path}")

    except Exception as e:
        logging.warning(f"⚠️ 描述性统计输出失败: {e}")


def save_reports(output_dir: Path, reports: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "pipeline_reports.json"

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

def filter_plan(plan, start_step: str | None = None, end_step: str | None = None, only_step: str | None = None):
    """
    根据命令行参数过滤执行计划：
    1. --only S5.6：只跑某一步
    2. --start S5.6：从某一步跑到最后
    3. --start S5.6 --end S6：从某一步跑到某一步
    """
    step_ids = [s.step_id for s in plan]

    if only_step:
        if only_step not in step_ids:
            raise ValueError(f"Unknown step_id: {only_step}. Available: {step_ids}")
        return [s for s in plan if s.step_id == only_step]

    start_idx = 0
    end_idx = len(plan) - 1

    if start_step:
        if start_step not in step_ids:
            raise ValueError(f"Unknown start_step: {start_step}. Available: {step_ids}")
        start_idx = step_ids.index(start_step)

    if end_step:
        if end_step not in step_ids:
            raise ValueError(f"Unknown end_step: {end_step}. Available: {step_ids}")
        end_idx = step_ids.index(end_step)

    if start_idx > end_idx:
        raise ValueError(f"start_step must be before end_step: {start_step} > {end_step}")

    return plan[start_idx:end_idx + 1]

DEFAULT_CONFIG = {
    "project_root": ".",
    "output_dir": "outputs",

    "taobao_raw_dir": "data/taobao_reviews",
    "ozon_raw_dir": "data/ozon_reviews",
    "taobao_homepage": "data/taobao_homepage.xlsx",
    "ozon_homepage": "data/ozon_homepage.xlsx",
    "oiq_llm_result": "data/oiq_llm_results.xlsx",

    "cn_base_model": "hfl/chinese-roberta-wwm-ext",
    "ru_base_model": "DeepPavlov/rubert-base-cased",
    "cn_dapt_output": "outputs/cn_dapt_model",
    "ru_dapt_output": "outputs/ru_expert_model",
    "cn_sentiment_model": "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
    "ru_sentiment_model": "cointegrated/rubert-tiny-sentiment-balanced",
    "clip_model": "openai/clip-vit-base-patch32",

    "taobao_home_img_dir": "data/taobao_home_images",
    "taobao_home_video_dir": "data/taobao_home_videos",
    "ozon_home_img_dir": "data/ozon_home_images",
    "ozon_home_video_dir": "data/ozon_home_videos",
    "taobao_review_img_dir": "data/taobao_review_images",
    "ozon_review_img_dir": "data/ozon_review_images",
}


def load_config_file(config_path: str | None = None) -> Dict[str, Any]:
    """
    Load non-public local configuration.

    Priority:
    1. CLI argument: --config path/to/config.local.json
    2. Environment variable: SOP_AGENT_CONFIG
    3. DEFAULT_CONFIG in this file

    Do not commit config.local.json to GitHub.
    """
    selected_path = config_path or os.getenv("SOP_AGENT_CONFIG")
    if not selected_path:
        return {}

    path = Path(selected_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_value(config: Dict[str, Any], key: str) -> Any:
    env_key = f"SOP_AGENT_{key.upper()}"
    if env_key in os.environ:
        return os.environ[env_key]
    return config.get(key, DEFAULT_CONFIG.get(key))


def _resolve_path(value: Any, project_root: Path, allow_none: bool = False) -> Path | None:
    if value is None or value == "":
        if allow_none:
            return None
        raise ValueError("Path value cannot be empty.")

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def _resolve_model_ref(value: Any) -> str | Path | None:
    """
    Keep HuggingFace model IDs as strings, while allowing local model paths.
    Examples:
    - "hfl/chinese-roberta-wwm-ext" -> remote/model-cache ID
    - "./models/my_model" -> local path string
    - "/absolute/path/to/models/my_model" -> local path string
    """
    if value is None or value == "":
        return None
    return str(value)


def _looks_like_hf_model_id(value: Any) -> bool:
    if value is None:
        return False
    s = str(value)
    return (
        "/" in s
        and not Path(s).is_absolute()
        and ":" not in s
        and not s.startswith(".")
        and not s.startswith("~")
    )


def build_config(config_path: str | None = None) -> LocalPaths:
    """
    Build local path configuration without hard-coding private machine paths.

    Recommended:
    1. Copy config.example.json to config.local.json
    2. Fill in your own local paths
    3. Run: python -m sop_agent_v4.main --config config.local.json
    """
    config = load_config_file(config_path)

    project_root_value = os.getenv(
        "SOP_AGENT_PROJECT_ROOT",
        config.get("project_root", DEFAULT_CONFIG["project_root"]),
    )
    project_root = Path(str(project_root_value)).expanduser()
    if not project_root.is_absolute():
        project_root = Path.cwd() / project_root
    project_root = project_root.resolve()

    output_dir = _resolve_path(_get_value(config, "output_dir"), project_root)

    cfg = LocalPaths(
        taobao_raw_dir=_resolve_path(_get_value(config, "taobao_raw_dir"), project_root),
        ozon_raw_dir=_resolve_path(_get_value(config, "ozon_raw_dir"), project_root),
        output_dir=output_dir,
        taobao_homepage=_resolve_path(_get_value(config, "taobao_homepage"), project_root, allow_none=True),
        ozon_homepage=_resolve_path(_get_value(config, "ozon_homepage"), project_root, allow_none=True),
        oiq_llm_result=_resolve_path(_get_value(config, "oiq_llm_result"), project_root, allow_none=True),

        cn_base_model=_resolve_model_ref(_get_value(config, "cn_base_model")),
        ru_base_model=_resolve_model_ref(_get_value(config, "ru_base_model")),
        cn_dapt_output=_resolve_path(_get_value(config, "cn_dapt_output"), project_root, allow_none=True),
        ru_dapt_output=_resolve_path(_get_value(config, "ru_dapt_output"), project_root, allow_none=True),
        cn_sentiment_model=_resolve_model_ref(_get_value(config, "cn_sentiment_model")),
        ru_sentiment_model=_resolve_model_ref(_get_value(config, "ru_sentiment_model")),
        clip_model=_resolve_model_ref(_get_value(config, "clip_model")),

        taobao_home_img_dir=_resolve_path(_get_value(config, "taobao_home_img_dir"), project_root, allow_none=True),
        taobao_home_video_dir=_resolve_path(_get_value(config, "taobao_home_video_dir"), project_root, allow_none=True),
        ozon_home_img_dir=_resolve_path(_get_value(config, "ozon_home_img_dir"), project_root, allow_none=True),
        ozon_home_video_dir=_resolve_path(_get_value(config, "ozon_home_video_dir"), project_root, allow_none=True),
        taobao_review_img_dir=_resolve_path(_get_value(config, "taobao_review_img_dir"), project_root, allow_none=True),
        ozon_review_img_dir=_resolve_path(_get_value(config, "ozon_review_img_dir"), project_root, allow_none=True),
    )

    return cfg

def validate_config(cfg: LocalPaths) -> None:
    """
    启动前检查关键路径是否存在。
    不直接中断全部流程，只写日志提醒。
    """
    checks = {
        "taobao_raw_dir": cfg.taobao_raw_dir,
        "ozon_raw_dir": cfg.ozon_raw_dir,
        "taobao_homepage": cfg.taobao_homepage,
        "ozon_homepage": cfg.ozon_homepage,
        "oiq_llm_result": cfg.oiq_llm_result,
        "cn_base_model": cfg.cn_base_model,
        "ru_base_model": cfg.ru_base_model,
        "clip_model": cfg.clip_model,
        "taobao_home_img_dir": cfg.taobao_home_img_dir,
        "ozon_home_img_dir": cfg.ozon_home_img_dir,
        "taobao_review_img_dir": cfg.taobao_review_img_dir,
        "ozon_review_img_dir": cfg.ozon_review_img_dir,
    }

    for name, path in checks.items():
        if path is None:
            logging.warning(f"⚠️ 配置缺失: {name}=None")
            continue

        if name in {"cn_base_model", "ru_base_model", "clip_model"} and _looks_like_hf_model_id(path):
            logging.info(f"ℹ️ HuggingFace 模型引用: {name} -> {path}")
            continue

        path = Path(path)

        if path.exists():
            logging.info(f"✅ 路径存在: {name} -> {path}")
        else:
            logging.warning(f"⚠️ 路径不存在: {name} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SOP Agent pipeline with checkpoint resume control.")

    parser.add_argument("--start", type=str, default=None, help="Start from a specific step, e.g. S5.6")
    parser.add_argument("--end", type=str, default=None, help="End at a specific step, e.g. S6")
    parser.add_argument("--only", type=str, default=None, help="Run only one step, e.g. S5.6")
    parser.add_argument("--config", type=str, default=None, help="Path to a local JSON config file. Do not commit this file.")

    args = parser.parse_args()

    cfg = build_config(args.config)
    setup_logging(cfg.output_dir)
    validate_config(cfg)

    logger = logging.getLogger("main")
    logger.info("🚀 启动 SOP Agent v4 多源异构数据融合流程")

    memory = AgentMemory()
    tools = LocalTools(cfg)
    planner = SOPPlanner()
    executor = SOPExecutor(tools=tools, memory=memory)

    try:
        plan = planner.build_plan()
        plan = filter_plan(plan, start_step=args.start, end_step=args.end, only_step=args.only)

        logger.info(f"📌 本次执行步骤: {[s.step_id for s in plan]}")

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

        memory.export_audit_report(cfg.output_dir)
        memory.save_research_manifest(str(cfg.output_dir / "research_manifest.json"))

        logger.info("🎉 流程执行完成")

    except Exception as e:
        logger.exception(f"❌ 主流程执行失败: {e}")
        raise


if __name__ == "__main__":
    main()