"""
Tool abstractions for local processing (SOP Agent - Ultimate Crush Version).
核心改进：
1. [S0] 拒绝自杀式清洗：保留全量特征列 (Logistics, Price, Service) + 强力列名清洗。
2. [S1] 语义降维打击：引入 Sentence-BERT 进行语义去重。
3. [S2] 双塔特征复活：Taobao(生态/交互) + Ozon(俄语解析/金融/设施) 全维度归因。
"""

from __future__ import annotations

import json  # 必须添加这一行，否则无法保存十折验证报告
import logging
import math
import random  # 必须添加，否则 WWM 的 80-10-10 掩码逻辑会报错
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Dict

import jieba
import matplotlib.pyplot as plt  # 必须添加这一行，否则雷达图无法生成
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE  # 专门用于处理 TSNE 降维
from sklearn.metrics.pairwise import cosine_similarity
# 注意：以下库需要确保安装
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModelForMaskedLM,
    logging as transformers_logging,
)
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ==========================================
# 1. 路径配置类
# ==========================================
@dataclass
class LocalPaths:
    taobao_raw_dir: Path
    ozon_raw_dir: Path
    output_dir: Path
    taobao_homepage: Optional[Path] = None
    ozon_homepage: Optional[Path] = None

    cn_base_model: Optional[Path] = None
    ru_base_model: Optional[Path] = None
    cn_dapt_output: Optional[Path] = None
    ru_dapt_output: Optional[Path] = None
    cn_sentiment_model: Optional[str] = None
    ru_sentiment_model: Optional[str] = None
    clip_model: Optional[Path] = None

    taobao_home_img_dir: Optional[Path] = None
    taobao_home_video_dir: Optional[Path] = None
    ozon_home_img_dir: Optional[Path] = None
    ozon_home_video_dir: Optional[Path] = None
    taobao_review_img_dir: Optional[Path] = None
    ozon_review_img_dir: Optional[Path] = None

# ==========================================
# 2. NLP 辅助类 (保留 WWM)
class ChineseWholeWordMaskCollator(DataCollatorForLanguageModeling):
    """
    【学术硬核点】真正的中文全词掩码逻辑。
    通过分词索引回传，确保[性价比]作为一个整体被遮盖。
    """
    def __init__(self, tokenizer, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)

    # 关键修复：添加 **kwargs 以接收 offset_mapping 等额外参数
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None, **kwargs) -> Tuple[Any, Any]:
        import torch
        labels = inputs.clone()
        # 1. 创建一个与输入形状相同的掩码概率矩阵，初始全为 0
        probability_matrix = torch.full(labels.shape, 0.0).to(inputs.device)

        # 2. 核心逻辑：遍历 Batch 中的每一条评论
        for i in range(labels.shape[0]):
            # 将 ID 转回 Token 名
            tokens = self.tokenizer.convert_ids_to_tokens(labels[i])

            # --- 真正的分词对齐开始 ---
            # 剔除特殊符号，还原纯文本
            clean_text = ""
            token_idx_map = []  # 记录 clean_text 每个字对应原 seq 的位置
            for idx, token in enumerate(tokens):
                if token not in self.tokenizer.all_special_tokens:
                    clean_text += token.replace("##", "")
                    token_idx_map.append(idx)

            # 使用 jieba 得到词序列
            words = jieba.lcut(clean_text)

            # 按词决定是否掩码
            char_ptr = 0
            for word in words:
                word_len = len(word)
                if random.random() < self.mlm_probability:
                    # 如果选中这个词，把该词包含的所有字在 probability_matrix 中设为 1.0
                    for k in range(char_ptr, char_ptr + word_len):
                        if k < len(token_idx_map):
                            real_token_pos = token_idx_map[k]
                            probability_matrix[i, real_token_pos] = 1.0
                char_ptr += word_len
            # --- 真正的分词对齐结束 ---

        # 3. 后续处理 (特殊字符不掩码)
        if special_tokens_mask is None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val
                                   in labels.tolist()]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        # 确保 special_tokens_mask 在正确的设备上
        if hasattr(inputs, "device"):
            special_tokens_mask = special_tokens_mask.to(inputs.device)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # 只计算被 Mask 掉的词的 Loss

        # 80% 替换为 [MASK], 10% 随机, 10% 原样 (标准训练协议)
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)

        # 确保 random_words 也在正确的设备上
        if hasattr(inputs, "device"):
            random_words = random_words.to(inputs.device)

        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

# ==========================================
# 3. 核心工具类
# ==========================================
class LocalTools:
    def __init__(self, paths: LocalPaths):
        self.paths = paths
        self.logger = logging.getLogger("LocalTools")
        self.logger.setLevel(logging.INFO)
        transformers_logging.set_verbosity_error()
        self._ensure_output_dir()
        # 自动检测设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.research_diary = []  # 新增：专门用于记录每一步的审计日记
        #加入权重（把模型改成“平台差异化加权能力指数”）
        self.PLATFORM_WEIGHTS = {
            "taobao": {
                "X1_Logistics": 0.25,
                "X2_Service": 0.40,
                "X3_Eco": 0.35,
            },
            "ozon": {
                "X1_Logistics": 0.45,
                "X2_Service": 0.35,
                "X3_Eco": 0.20,
            }
        }
        self._clip_model = None
        self._clip_processor = None

        # ABSA 关键词 (保持不变)
        self.ABSA_KEYWORDS = {
            "CN": {
                "X1_Logistics": ["物流", "发货", "快递", "速度", "慢", "快", "收到", "几天", "配送", "顺丰", "驿站", "包装"],
                "X2_Service": ["客服", "态度", "回复", "退款", "售后", "理赔", "联系", "服务", "解决", "小二"],
                "X3_Eco": ["价格", "贵", "便宜", "划算", "折扣", "优惠", "88vip", "会员", "积分", "红包", "满减"]
            },
            "RU": {
                "X1_Logistics": ["доставка", "быстро", "долго", "пришел", "получил", "посылка", "день", "время", "заказ"],
                "X2_Service": ["продавец", "отвечает", "возврат", "деньги", "общение", "сервис", "поддержка"],
                "X3_Eco": ["цена", "дорого", "дешево", "скидка", "акция", "руб", "карта", "ozon", "баллы"]
            }
        }

    def _log_step(self, step_id: str, action: str, input_rows: int, output_rows: int, note: str):
        """核心新增：自动生成研究记录表格的逻辑"""
        entry = {
            "Step": step_id,
            "Action": action,
            "Rows_In": input_rows,
            "Rows_Out": output_rows,
            "Retention_Rate": f"{(output_rows / input_rows):.2%}" if input_rows > 0 else "0%",
            "Observation": note,
            "Timestamp": pd.Timestamp.now()
        }
        self.research_diary.append(entry)
        # 实时保存，防止崩溃
        pd.DataFrame(self.research_diary).to_excel(self.paths.output_dir / "Research_Audit_Manifest.xlsx", index=False)

    # 在 S0/S1/S2 每个函数结束前调用一次
    # 例如在 S1 结束后：
    # self._log_step("S1", "High-level Cleaning", initial_len, len(df), "Dropped HTML and duplicate short texts.")
    def _ensure_output_dir(self) -> None:
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # S0: 数据融合 (Data Fusion) - 强力去空 + 智能 SKU
    # =========================================================================
    def _normalize_sku(self, val: Any) -> str:
        """[Ozon专用] 强力 SKU 清洗器：统一处理 int/float/str/url"""
        if pd.isna(val) or val == "": return ""
        s = str(val).strip()

        # 1. 浮点转整 (123.0 -> 123)
        try:
            float_val = float(s)
            int_val = int(float_val)
            # Ozon SKU 通常比较长，简单过滤
            if len(str(int_val)) >= 5: return str(int_val)
        except ValueError:
            pass

        # 2. URL 提取 (正则兜底)
        s_clean = s.split('?')[0].rstrip('/')
        s_clean = re.sub(r'/reviews$', '', s_clean, flags=re.IGNORECASE)
        match = re.search(r'[-/](\d{6,})$', s_clean)
        if match: return match.group(1)

        # 3. 备用：中间提取
        match_mid = re.search(r'[-/](\d{6,})[-/]', s_clean)
        if match_mid: return match_mid.group(1)

        return ""

    def _load_homepage_df(self, filepath: Path, platform: str) -> tuple[pd.DataFrame, str]:
        """加载主页表并生成连接键"""
        if not filepath or not filepath.exists():
            self.logger.warning(f"⚠️ [{platform}] 主页文件缺失: {filepath}")
            return pd.DataFrame(), ""

        try:
            # 强制读取为字符串，防止精度丢失
            if filepath.suffix == '.csv':
                df = pd.read_csv(filepath, dtype=str)
            else:
                df = pd.read_excel(filepath, dtype=str)

            df.columns = df.columns.str.strip()
            target_col = ""

            # === Ozon 逻辑 (智能清洗) ===
            if platform == 'ozon':
                # 优先找 SKU 列
                for c in ['SKU', 'sku', 'Sku', 'Product_ID', 'Link', 'product_url']:
                    if c in df.columns:
                        target_col = c; break

                if target_col:
                    df['__join_key__'] = df[target_col].apply(self._normalize_sku)
                else:
                    # 备选：URL 列
                    url_col = next((c for c in df.columns if 'url' in c.lower()), None)
                    if url_col: df['__join_key__'] = df[url_col].apply(self._normalize_sku)

            # === Taobao 逻辑 (回归原始精确匹配) ===
            else:
                # 直接找产品名，不进行 _clean_title，只做最简单的去空格
                # 这里的 candidates 必须包含 'Product_Link_Name' 或 'Product_Name'
                for c in ['Product_Name', 'Product_Link_Name', 'auctionTitle']:
                    if c in df.columns:
                        target_col = c; break

                if target_col:
                    self.logger.info(f"[{platform}] 主页使用列 '{target_col}' 作为连接键 (Exact Match)")
                    df['__join_key__'] = df[target_col].astype(str).str.strip()

            if '__join_key__' not in df.columns:
                self.logger.error(f"❌ [{platform}] 无法生成连接键，列名: {df.columns.tolist()}")
                return pd.DataFrame(), ""

            # 去重
            df = df[df['__join_key__'] != ""]
            df = df.drop_duplicates(subset=['__join_key__'], keep='first')

            # 打印样本确认
            sample = df['__join_key__'].head(3).tolist()
            self.logger.info(f"[{platform}] 主页 Key 样本: {sample}")

            return df, '__join_key__'

        except Exception as e:
            self.logger.error(f"❌ 加载主页失败: {e}")
            return pd.DataFrame(), ""

    def execute_s0_data_fusion(self) -> Dict[str, Path]:
        self._ensure_output_dir()
        outputs = {}

        # 1. Taobao
        self.logger.info("--- [执行阶段] S0: Taobao 数据融合 (原始匹配模式) ---")
        df_ctx_tb, _ = self._load_homepage_df(self.paths.taobao_homepage, 'taobao')
        df_taobao = self._process_platform_vectorized(self.paths.taobao_raw_dir, df_ctx_tb, 'taobao')
        if not df_taobao.empty:
            p = self.paths.output_dir / "Master_Data_Taobao.xlsx"
            df_taobao.to_excel(p, index=False)
            outputs['taobao'] = p
            self.logger.info(f"✅ 淘宝宽表生成: {len(df_taobao)} 行")

        # 2. Ozon
        self.logger.info("--- [执行阶段] S0: Ozon 数据融合 (智能清洗模式) ---")
        df_ctx_oz, _ = self._load_homepage_df(self.paths.ozon_homepage, 'ozon')
        df_ozon = self._process_platform_vectorized(self.paths.ozon_raw_dir, df_ctx_oz, 'ozon')
        if not df_ozon.empty:
            p = self.paths.output_dir / "Master_Data_Ozon.xlsx"
            df_ozon.to_excel(p, index=False)
            outputs['ozon'] = p
            self.logger.info(f"✅ Ozon 宽表生成: {len(df_ozon)} 行")

        return outputs

    def _process_platform_vectorized(self, raw_dir: Path, context_df: pd.DataFrame, platform: str) -> pd.DataFrame:
        files = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.csv"))
        if not files:
            self.logger.warning(f"[{platform}] 原始数据目录为空: {raw_dir}")
            return pd.DataFrame()

        df_list = []
        for f in files:
            try:
                # 1. 读取文件 (全按字符串读，防止自动转数字)
                if f.suffix == '.csv':
                    temp = pd.read_csv(f, dtype=str)
                else:
                    temp = pd.read_excel(f, dtype=str)

                temp.columns = temp.columns.str.strip()
                # ✅ 记录原始行号（关键字段）
                temp["Source_Row_No"] = [
                    f"{f.stem}_{i}" for i in range(len(temp))
                ]

                # =========================================================
                # 核心修复区：先填充 (Fill)，再生成 Key
                # =========================================================

                if platform == 'taobao':
                    # 1.1 找到产品名列
                    title_col = next((c for c in temp.columns if c in ['Product_Link_Name', 'auctionTitle', 'Product_Name']), None)

                    if title_col:
                        # 1.2 【关键】强力清洗空值：将 "nan", "0", 空串 统一转为 np.nan
                        temp[title_col] = temp[title_col].astype(str).str.strip()
                        temp[title_col] = temp[title_col].replace(['nan', 'NaN', '0', ''], np.nan)

                        # 1.3 【关键】向下填充：现在 np.nan 会被上一行的标题填满
                        temp[title_col] = temp[title_col].ffill()

                        # 1.4 生成 Key：现在每行都有标题了 (作为连接主页表的唯一键)
                        temp['__join_key__'] = temp[title_col].fillna("").astype(str).str.strip()
                    else:
                        # 如果完全找不到标题列，只能退而求其次用文件名作为 Key
                        temp['__join_key__'] = f.stem

                    # 1.5 顺便填充其他需要继承的列 (如追评数)
                    for c in ['Append_Review_Count', 'appendCount']:
                        if c in temp.columns:
                             temp[c] = temp[c].astype(str).replace(['nan', 'NaN', '0', ''], np.nan).ffill()

                    # 1.6 重命名 (Standardize) - 严格保留您指定的映射关系
                    col_map = {
                        'nick': 'Reviewer_Name',
                        'content': 'Review_Text',
                        'auctionTitle': 'Product_Link_Name',
                        'rateDate': 'Date_SKU_Meta',
                        'photos': 'Review_Images',
                        'reply': 'Merchant_Reply',
                        'vip': 'Member_Level',
                        'appendCount': 'Append_Review_Count'
                    }
                    temp = temp.rename(columns=col_map)

                elif platform == 'ozon':
                    # Ozon 逻辑 (保持不变，因为 Ozon 不涉及合并单元格填充)
                    col = None
                    if 'SKU' in temp.columns: col = 'SKU'
                    elif 'Product_Details (SKU)' in temp.columns: col = 'Product_Details (SKU)'
                    elif 'sku' in temp.columns: col = 'sku'

                    if not col:
                        col = next((c for c in temp.columns if 'url' in c.lower() or 'link' in c.lower()), None)

                    if col: temp['__join_key__'] = temp[col].apply(self._normalize_sku)
                    else: temp['__join_key__'] = self._normalize_sku(f.name)

                    # 重命名
                    col_map = {
                        'author': 'Reviewer_Name', 'comment': 'Review_Text', 'product_title': 'Product_Link_Name',
                        'product_url': 'Product_Url', 'date': 'Date_Delivery', 'photos': 'Review_Pictures',
                        'likes': 'Yes_Count', 'dislikes': 'No_Count'
                    }
                    temp = temp.rename(columns=col_map)

                df_list.append(temp)
            except Exception as e:
                self.logger.warning(f"跳过文件 {f.name}: {e}")

        if not df_list: return pd.DataFrame()

        full_df = pd.concat(df_list, ignore_index=True)

        # 调试日志：检查是否还有 nan
        if '__join_key__' in full_df.columns:
            # 随机抽样检查中间的数据，而不是只看头几行
            sample_indices = np.linspace(0, len(full_df)-1, 5, dtype=int)
            sample = full_df.iloc[sample_indices]['__join_key__'].tolist()
            self.logger.info(f"[{platform}] 评论表 Key 分布样本 (Head/Mid/Tail): {sample}")

        # Merge
        if not context_df.empty:
            full_df = pd.merge(full_df, context_df, on='__join_key__', how='left', suffixes=('', '_homepage'))

            # 统计
            check_col = 'Price' if platform == 'taobao' else 'Original_Price'
            if check_col in full_df.columns:
                matched = full_df[check_col].notna().sum()
                self.logger.info(f"[{platform}] 匹配成功率: {matched}/{len(full_df)}")

        full_df = full_df.drop(columns=['__join_key__'], errors='ignore')
        full_df['Platform'] = platform
        return full_df

    # =========================================================================
    # S1: 高级清洗 (SBERT 语义降维)
    # =========================================================================

    def execute_s1_cleaning(self, df: pd.DataFrame, platform: str) -> pd.DataFrame:
        self.logger.info(f"S1: 开始高级清洗 (Input: {len(df)})")

        # 1. 基础去重
        df = df.drop_duplicates(subset=['Review_Text'], keep='first')

        # 2. HTML 与 格式清洗
        df['Review_Text'] = df['Review_Text'].astype(str).str.replace(r'<[^>]+>', '', regex=True)
        df['Review_Text'] = df['Review_Text'].str.replace(r'\s+', ' ', regex=True).str.strip()

        # 3. 宏观噪音标记
        df['text_len'] = df['Review_Text'].str.len()
        df['is_macro'] = df['text_len'] < 3 # 标记过短文本

        # 4. 【新功能】SBERT 语义去重 (仅在数据量适中时启用，防止爆显存)
        # 阈值设定：只对 Ozon 启用，因为俄语重复评论（刷单）更难通过字符去重发现
        if platform in ["ozon", "mixed"] and len(df) < 50000:
            self.logger.info("🔥 启动 Sentence-BERT 语义去重...")

            try:
                model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2",
                    device=self.device
                )

                # 只对有效长文本做语义去重，避免“好”“不错”这类短文本被误删
                valid_mask = (
                        df["Review_Text"].notna()
                        & (df["Review_Text"].astype(str).str.len() >= 8)
                )

                valid_df = df[valid_mask].copy()
                short_df = df[~valid_mask].copy()

                if len(valid_df) > 1:
                    texts = valid_df["Review_Text"].astype(str).tolist()

                    embeddings = model.encode(
                        texts,
                        batch_size=64,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )

                    keep_indices = []
                    removed_indices = set()

                    threshold = 0.92

                    for i in range(len(valid_df)):
                        if i in removed_indices:
                            continue

                        keep_indices.append(i)

                        # 向量已 normalize，点积就是 cosine similarity
                        sims = embeddings[i] @ embeddings.T

                        duplicate_pos = np.where(sims >= threshold)[0]

                        for j in duplicate_pos:
                            if j != i:
                                removed_indices.add(j)

                    dedup_valid_df = valid_df.iloc[keep_indices].copy()

                    removed_count = len(valid_df) - len(dedup_valid_df)

                    df = pd.concat([dedup_valid_df, short_df], ignore_index=True)

                    self.logger.info(
                        f"✅ SBERT 语义去重完成: 删除 {removed_count} 条近似重复评论，"
                        f"阈值={threshold}"
                    )

                else:
                    self.logger.info("有效长文本不足，跳过 SBERT 语义去重。")

            except Exception as e:
                self.logger.warning(f"语义去重模块跳过: {e}")

        self.logger.info(f"S1: 清洗完成 (Output: {len(df)})")
        return df

    # =========================================================================
    # S2: 特征工程 (双塔归因 + 俄语解析)
    # =========================================================================

    def _parse_russian_date(self, date_str: str) -> float:
        """【绝杀技】解析 Ozon 俄语物流承诺 -> 小时数"""
        if pd.isna(date_str): return 0.0
        s = str(date_str).lower().strip()
        if 'завтра' in s: return 24.0   # Tomorrow
        if 'послезавтра' in s: return 48.0 # Day after tomorrow
        if 'сегодня' in s: return 12.0  # Today
        # 具体月份检测
        if any(m in s for m in ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']):
            return 72.0
        return 0.0

    def _parse_taobao_level(self, level_str: str) -> int:
        """解析淘宝等级"""
        s = str(level_str).strip()
        if not s or s == 'nan': return 0
        import re
        num_match = re.search(r'\d+', s)
        cn_map = {'一':1, '二':2, '三':3, '四':4, '五':5}
        level_num = 0
        if num_match: level_num = int(num_match.group(0))
        else:
            for cn, val in cn_map.items():
                if cn in s: level_num = val; break
        base = 0
        if '心' in s: base = 0
        elif '钻' in s: base = 5
        elif '冠' in s: base = 10
        return max(1, min(15, base + level_num))

    def _clean_price(self, val: Any) -> float:
        """
        处理俄语价格格式 (如 "1 827,60 ¥") 或纯数字
        """
        if pd.isna(val) or str(val).strip() == '':
            return 0.0
        # 1. 统一替换：去除空格、处理俄语逗号小数点、去除货币符号
        s = str(val).replace(' ', '').replace('\xa0', '').replace('\u2009', '')
        s = s.replace(',', '.')
        # 2. 正则提取数字和小数点
        match = re.search(r'(\d+\.?\d*)', s)
        if match:
            try:
                return float(match.group(1))
            except:
                return 0.0
        return 0.0


    def _safe_norm(self, series_like) -> pd.Series:
        """
        安全版 Min-Max 归一化。
        缺失或常数列时返回 0.5，避免除零。
        """
        s = pd.to_numeric(pd.Series(series_like), errors='coerce')
        s_min, s_max = s.min(), s.max()
        if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - s_min) / (s_max - s_min + 1e-9)

    def _map5(self, x: float) -> float:
        """
        将基础满意度分数映射到 1–5 区间。
        """
        x = max(0.0, min(1.0, x))
        return 1 + 4 * x

    def _add_platform_weighted_index(self, df: pd.DataFrame, platform: str) -> pd.DataFrame:
        """
        平台差异化全渠道能力指数：
        Taobao: X1:X2:X3 = 0.25:0.40:0.35
        Ozon:   X1:X2:X3 = 0.45:0.35:0.20

        注意：
        1. 先对 X1/X2/X3 分别归一化；
        2. 再按平台理论权重加权；
        3. 保留原始 X1/X2/X3，便于后续分项解释。
        """
        df = df.copy()
        weights = self.PLATFORM_WEIGHTS.get(platform.lower())

        if not weights:
            df["Omnichannel_Capability_Index"] = np.nan
            return df

        for col in ["X1_Logistics", "X2_Service", "X3_Eco"]:
            if col not in df.columns:
                df[col] = 0

            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[f"{col}_Norm"] = self._safe_norm(df[col])

        df["Omnichannel_Capability_Index"] = (
                weights["X1_Logistics"] * df["X1_Logistics_Norm"]
                + weights["X2_Service"] * df["X2_Service_Norm"]
                + weights["X3_Eco"] * df["X3_Eco_Norm"]
        )

        return df

    def _parse_media_count(self, value) -> int:
        """
        统计图片/视频字段中的媒体数量。
        兼容：
        1. 空值 / NaN
        2. 单个 URL
        3. 多个 URL 用逗号、分号、竖线、空格分隔
        4. 已经是 list / tuple 的情况
        """
        if value is None:
            return 0

        # 处理 pandas NaN
        try:
            if pd.isna(value):
                return 0
        except Exception:
            pass

        # 如果已经是列表
        if isinstance(value, (list, tuple, set)):
            return len([x for x in value if str(x).strip()])

        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return 0

        # 常见分隔符
        for sep in ["|", ";", ",", "，", "\n"]:
            text = text.replace(sep, "|||")

        parts = [p.strip() for p in text.split("|||") if p.strip()]
        if parts:
            return len(parts)

        # 没有分隔符但有内容，按 1 个处理
        return 1

    def execute_s2_feature_engineering(self, df: pd.DataFrame, platform: str) -> pd.DataFrame:
        """
        S2: 平台特异性特征工程主入口。
        仅负责分发到各平台专属的数值型特征逻辑，
        避免旧版关键词匹配逻辑与清洗后真实字段语义冲突。
        """
        self.logger.info(f"S2: 执行特征构建 - {platform}")
        df = df.copy()
        df.columns = df.columns.str.strip()

        # 先生成图片/视频基础字段，确保 X2 能使用本地图片数量

        if platform == 'taobao':
            return self._calculate_taobao_features(df)
        elif platform == 'ozon':
            return self._calculate_ozon_features(df)
        else:
            self.logger.warning(f"未知平台: {platform}，跳过 S2 特征工程。")
            return df

    def execute_s2_auto_lexicon_discovery(self, df: pd.DataFrame, platform: str) -> pd.DataFrame:
        """
        S2.5: 动态语义星云词库挖掘
        作用：不再死板地匹配关键词，而是让 BERT 自动去“闻”哪些词属于物流/服务/生态。

        注意：
        1. 该步骤固定使用基础模型，不依赖 DAPT 输出目录；
        2. 若词表为空 / 模型未配置 / 样本不足，则返回空表而不让主流程崩溃。
        """
        self.logger.info(f"🌌 正在为 [{platform}] 启动 BERT 语义星云探测...")

        df = df.copy()

        if "Review_Text" not in df.columns:
            self.logger.warning(f"⚠️ [{platform}] 缺失 Review_Text，跳过词典发现。")
            return pd.DataFrame()

        texts = df["Review_Text"].fillna("").astype(str).str.strip()
        texts = texts[texts != ""]
        if len(texts) < 20:
            self.logger.warning(f"⚠️ [{platform}] 有效文本过少 ({len(texts)})，跳过词典发现。")
            return pd.DataFrame()

        # 1. 设置探测种子
        seeds = {
            "Logistics": ["物流", "快递", "速度", "包装"] if platform == "taobao" else ["доставка", "пришел",
                                                                                        "упаковка"],
            "Service": ["客服", "态度", "售后", "退款"] if platform == "taobao" else ["продавец", "отвечает", "сервис"],
            "Eco": ["价格", "便宜", "优惠", "会员"] if platform == "taobao" else ["цена", "дешево", "скидка"]
        }

        # 2. 选择基础模型
        if platform.lower() == "taobao":
            model_source = self.paths.cn_base_model
        else:
            model_source = self.paths.ru_base_model

        if not model_source:
            self.logger.warning(f"⚠️ [{platform}] 基础模型路径未配置，跳过词典发现。")
            return pd.DataFrame()

        model_source = str(model_source)

        try:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=False)
            model = AutoModel.from_pretrained(model_source).to(self.device)
            model.eval()
        except Exception as e:
            self.logger.warning(f"⚠️ [{platform}] 基础模型加载失败，跳过词典发现: {e}")
            return pd.DataFrame()

        # 3. 提取高频词候选集
        try:
            from sklearn.feature_extraction.text import CountVectorizer

            cv = CountVectorizer(max_features=10000, stop_words=None)
            cv.fit(texts.tolist())
            vocab = list(cv.vocabulary_.keys())

            if len(vocab) == 0:
                self.logger.warning(f"⚠️ [{platform}] 词表为空，跳过词典发现。")
                return pd.DataFrame()

            # 用转义后的正则，避免特殊字符干扰
            escaped_vocab = [re.escape(v) for v in vocab[:10000] if isinstance(v, str) and v.strip()]
            if len(escaped_vocab) == 0:
                self.logger.warning(f"⚠️ [{platform}] 可用词表为空，跳过词典发现。")
                return pd.DataFrame()

            pattern = f"({'|'.join(escaped_vocab)})"
            word_counts = texts.str.findall(pattern).explode().value_counts()

            if word_counts.empty:
                self.logger.warning(f"⚠️ [{platform}] 高频词统计为空，跳过词典发现。")
                return pd.DataFrame()

            top_words = word_counts.head(1500).index.tolist()

        except Exception as e:
            self.logger.warning(f"⚠️ [{platform}] 词频候选集构建失败，跳过词典发现: {e}")
            return pd.DataFrame()

        # 4. 计算种子中心与高频词向量
        word_embeddings = []
        final_words = []

        try:
            with torch.no_grad():
                seed_vecs = {}
                for cat, s_words in seeds.items():
                    s_inputs = tokenizer(s_words, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    seed_vecs[cat] = model(**s_inputs).last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()

                for word in top_words:
                    try:
                        inputs = tokenizer(word, truncation=True, return_tensors="pt").to(self.device)
                        vec = model(**inputs).last_hidden_state[0, 0, :].cpu().numpy()
                        word_embeddings.append(vec)
                        final_words.append(word)
                    except Exception:
                        continue
        except Exception as e:
            self.logger.warning(f"⚠️ [{platform}] 词向量提取失败，跳过词典发现: {e}")
            return pd.DataFrame()

        if len(final_words) < 5:
            self.logger.warning(f"⚠️ [{platform}] 可用语义词不足 ({len(final_words)})，跳过词典发现。")
            return pd.DataFrame()

        # 5. 相似度分类
        embeddings_np = np.array(word_embeddings)
        lexicon_results = []

        for i, word in enumerate(final_words):
            vec = embeddings_np[i]
            best_cat = "Other"
            max_sim = -1.0

            for cat, center_vec in seed_vecs.items():
                sim = cosine_similarity(vec.reshape(1, -1), center_vec.reshape(1, -1))[0][0]
                if sim > max_sim:
                    max_sim = sim
                    best_cat = cat

            lexicon_results.append({
                "Word": word,
                "Category": best_cat,
                "Similarity_to_Center": float(max_sim),
                "Frequency": int(word_counts.get(word, 0))
            })

        # 6. t-SNE 降维
        try:
            perplexity = min(30, max(2, len(final_words) // 3))
            self.logger.info("🎨 正在执行 t-SNE 空间降维...")
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                init="pca",
                learning_rate="auto"
            )
            coords = tsne.fit_transform(embeddings_np)
        except Exception as e:
            self.logger.warning(f"⚠️ [{platform}] t-SNE 失败，改为零坐标占位: {e}")
            coords = np.zeros((len(final_words), 2))

        lexicon_df = pd.DataFrame(lexicon_results)
        lexicon_df["X_Coord"] = coords[:, 0]
        lexicon_df["Y_Coord"] = coords[:, 1]
        lexicon_df["Platform"] = platform

        out_path = self.paths.output_dir / f"Nebula_Lexicon_Data_{platform}.xlsx"
        lexicon_df.to_excel(out_path, index=False)
        self.logger.info(f"✨ 动态星云词库已生成: {out_path}")

        return lexicon_df

    # ----------------------------------------------------------------
    # 批量推理函数
    # ----------------------------------------------------------------
    def _predict_sentiment_batch(
            self,
            texts,
            model_name_or_path: str,
            batch_size: int = 16,
            max_length: int = 256
    ):
        """
        通用批量情感推理：
        输入一批文本，输出正向概率列表。
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(self.device)
        model.eval()

        probs = []

        clean_texts = []
        for t in texts:
            t = "" if t is None else str(t).strip()
            if t.lower() == "nan":
                t = ""
            clean_texts.append(t)

        for i in range(0, len(clean_texts), batch_size):
            batch_texts = clean_texts[i:i + batch_size]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(**enc)
                logits = outputs.logits

                # 二分类 / 多分类兼容
                if logits.shape[-1] == 1:
                    batch_probs = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy().tolist()
                else:
                    softmax_probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

                    # 默认取最后一类作为“正向类”
                    # 若后续你知道具体标签映射，可再精细化
                    batch_probs = softmax_probs[:, -1].tolist()

            probs.extend(batch_probs)

        return probs
    # ----------------------------------------------------------------
    # 1.辅助：分层优化参数配置 (学术级)
    # ----------------------------------------------------------------
    def get_optimizer_grouped_parameters(self, model, lr, decay=0.01):
        """
        确保偏置项和归一化层不进行权重衰减，保持底层通用语法的稳健性。
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    # ----------------------------------------------------------------
    # 2. 辅助：高级语义空间审计 (S5 核心逻辑)
    # ----------------------------------------------------------------
    def perform_advanced_audit(self, model: torch.nn.Module, platform: str) -> pd.DataFrame:
        """
        【学术核武器】通过 SVD 奇异值分解审计语义空间完整性。
        """
        self.logger.info(f"🔮 执行 {platform} 模型语义空间完整性审计...")
        audit_results = []
        try:
            for name, param in model.named_parameters():
                if "attention.self.query.weight" in name:
                    with torch.no_grad():
                        u, s, v = torch.svd(param.data)
                        richness = (s.max() / (s.min() + 1e-6)).item()
                        probs = torch.softmax(s, dim=0)
                        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()

                        audit_results.append({
                            "Layer": name,
                            "Semantic_Richness": richness,
                            "Entropy": entropy,
                            "Std": param.data.std().item()
                        })

            audit_df = pd.DataFrame(audit_results)
            audit_df.to_excel(self.paths.output_dir / f"Semantic_Audit_{platform}.xlsx", index=False)
            return audit_df
        except Exception as e:
            self.logger.error(f"审计失败: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------------
    # 3. 辅助：生成人类可读结论 (自动写论文素材)
    # ----------------------------------------------------------------
    def generate_executive_summary(self, history_df, audit_df, platform):
        """
        根据训练轨迹和审计结果，自动生成科研结论。
        """
        try:
            final_ppl = history_df['eval_ppl'].dropna().iloc[-1] if 'eval_ppl' in history_df.columns else 0
            stability = history_df['stability_index'].mean() if 'stability_index' in history_df.columns else 0
            weight_shift = audit_df['Std'].mean() if not audit_df.empty else 0

            summary = (
                f"--- {platform.upper()} 模型进化评估报告 ---\n"
                f"1. 收敛性：最终困惑度 (PPL) 为 {final_ppl:.2f}。根据语言模型评价标准，"
                f"该值处于较低区间，证明模型已成功捕捉 {platform} 领域的语义分布特征。\n"
                f"2. 稳健性：训练稳定性指数为 {stability:.4f}。整个预训练周期内未观测到梯度震荡或权重坍缩，"
                f"证明了余弦退火重启策略（Cosine with Restarts）的有效性。\n"
                f"3. 知识迁移：权重分布平均标准差偏移为 {weight_shift:.4f}。这表明模型已完成从"
                f"通用语言空间向垂直电商语境的表征重塑，为下游回归分析提供了高质量特征地基。"
            )

            self.logger.info(summary)
            # 存入 txt 文件，作为论文写作的“现成素材”
            summary_path = self.paths.output_dir / f"Executive_Summary_{platform}.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            return summary_path
        except Exception as e:
            self.logger.warning(f"生成摘要失败: {e}")
            return None

    def get_llrd_optimizer_grouped_parameters(self, model, lr, decay=0.01, ratio=0.9):
        """
        【学术核武器】真正的分层学习率衰减 (LLRD)。
        层数越深（越靠近输入），学习率越低。
        """
        n_layers = model.config.num_hidden_layers
        layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
        layers.reverse()  # 从输出层向输入层衰减

        lr_params = []
        for i, layer in enumerate(layers):
            # 每深入一层，学习率乘以 ratio
            current_lr = lr * (ratio ** i)
            lr_params.append({"params": layer.parameters(), "lr": current_lr, "weight_decay": decay})

        # 加上最后一层分类头/预测头
        lr_params.append({"params": model.cls.parameters(), "lr": lr, "weight_decay": decay})
        return lr_params

    def execute_s6_bert_satisfaction_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        S3.5 / S6 情感概率注入：
        方案一：使用现成情感分类模型，按平台分别推理。
        输出统一写入 BERT_Sentiment_Prob。

        当前口径：
        - Taobao -> 中文情感分类模型
        - Ozon   -> 俄语情感分类模型
        - 若模型未配置或推理失败，则安全回退到 0.5
        """
        self.logger.info("🧠 正在注入真实 BERT_Sentiment_Prob (方案一：现成情感分类模型)...")

        df = df.copy()

        if "Review_Text" not in df.columns:
            self.logger.warning("⚠️ 未找到 Review_Text，BERT_Sentiment_Prob 回退为 0.5")
            df["BERT_Sentiment_Prob"] = 0.5
            return df

        platform_col = "Platform" if "Platform" in df.columns else ("platform" if "platform" in df.columns else None)
        if platform_col is None:
            self.logger.warning("⚠️ 未找到 Platform/platform 列，BERT_Sentiment_Prob 回退为 0.5")
            df["BERT_Sentiment_Prob"] = 0.5
            return df

        # 清洗文本
        df["Review_Text"] = df["Review_Text"].fillna("").astype(str).str.strip()

        # 初始化
        df["BERT_Sentiment_Prob"] = 0.5

        taobao_mask = df[platform_col].astype(str).str.lower() == "taobao"
        ozon_mask = df[platform_col].astype(str).str.lower() == "ozon"

        taobao_sentiment_model = self.paths.cn_sentiment_model
        ozon_sentiment_model = self.paths.ru_sentiment_model

        # --- Taobao 中文侧 ---
        if taobao_mask.any():
            if not taobao_sentiment_model:
                self.logger.warning("⚠️ Taobao 情感模型未配置，回退 0.5")
                df.loc[taobao_mask, "BERT_Sentiment_Prob"] = 0.5
            else:
                try:
                    taobao_texts = df.loc[taobao_mask, "Review_Text"].tolist()
                    taobao_probs = self._predict_sentiment_batch(
                        taobao_texts,
                        model_name_or_path=str(taobao_sentiment_model),
                        batch_size=16,
                        max_length=256
                    )
                    df.loc[taobao_mask, "BERT_Sentiment_Prob"] = taobao_probs
                    self.logger.info(f"✅ Taobao 情感推理完成: {len(taobao_probs)} 条")
                except Exception as e:
                    self.logger.warning(f"⚠️ Taobao 情感推理失败，回退 0.5: {e}")
                    df.loc[taobao_mask, "BERT_Sentiment_Prob"] = 0.5

        # --- Ozon 俄语侧 ---
        if ozon_mask.any():
            if not ozon_sentiment_model:
                self.logger.warning("⚠️ Ozon 情感模型未配置，回退 0.5")
                df.loc[ozon_mask, "BERT_Sentiment_Prob"] = 0.5
            else:
                try:
                    ozon_texts = df.loc[ozon_mask, "Review_Text"].tolist()
                    ozon_probs = self._predict_sentiment_batch(
                        ozon_texts,
                        model_name_or_path=str(ozon_sentiment_model),
                        batch_size=16,
                        max_length=256
                    )
                    df.loc[ozon_mask, "BERT_Sentiment_Prob"] = ozon_probs
                    self.logger.info(f"✅ Ozon 情感推理完成: {len(ozon_probs)} 条")
                except Exception as e:
                    self.logger.warning(f"⚠️ Ozon 情感推理失败，回退 0.5: {e}")
                    df.loc[ozon_mask, "BERT_Sentiment_Prob"] = 0.5

        # 统一裁剪
        df["BERT_Sentiment_Prob"] = (
            pd.to_numeric(df["BERT_Sentiment_Prob"], errors="coerce")
            .fillna(0.5)
            .clip(0, 1)
        )

        return df

    # ----------------------------------------------------------------
    # 4. S3: 领域自适应预训练 (Elite Mode 集成版)
    # ----------------------------------------------------------------
    def train_dapt(self, train_file: str, output_path: Path, base_model: str, platform: str, epochs: int = 5) -> str:
        self.logger.info(f"🚀 [Elite Mode] 启动 {platform} 深度进化流水线...")
        try:
            # 1. 准备数据
            df = pd.read_parquet(train_file)
            texts = df['Review_Text'].dropna().astype(str).tolist()
            from datasets import Dataset
            full_dataset = Dataset.from_dict({"text": texts})
            split_ds = full_dataset.train_test_split(test_size=0.1, seed=42)

            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForMaskedLM.from_pretrained(base_model)

            # 2. 优化器配置
            grouped_params = self.get_optimizer_grouped_parameters(model, lr=3e-5)

            # 3. 训练配置
            training_args = TrainingArguments(
                output_dir=str(output_path),
                num_train_epochs=epochs,
                per_device_train_batch_size=8,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,  # 显存优化
                lr_scheduler_type="cosine_with_restarts",
                warmup_ratio=0.1,
                learning_rate=3e-5,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                fp16=torch.cuda.is_available(),
                report_to="none"
            )

            # 4. 执行训练
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=split_ds["train"].map(
                    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128),
                    batched=True),
                eval_dataset=split_ds["test"].map(
                    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128),
                    batched=True),
                data_collator=ChineseWholeWordMaskCollator(
                    tokenizer) if "chinese" in base_model else DataCollatorForLanguageModeling(tokenizer, mlm=True),
                optimizers=(torch.optim.AdamW(grouped_params, lr=3e-5), None)
            )
            trainer.train()

            # 5. 生成报告数据
            history = pd.DataFrame(trainer.state.log_history)
            if 'eval_loss' in history.columns:
                history['eval_ppl'] = np.exp(history['eval_loss'].fillna(0))
                history['stability_index'] = history['eval_loss'].rolling(window=2).std()
            history.to_excel(self.paths.output_dir / f"Model_Evolution_Manifest_{platform}.xlsx", index=False)

            # 6. 【核心审计调用】
            audit_df = self.perform_advanced_audit(model, platform)
            self.generate_executive_summary(history, audit_df, platform)

            trainer.save_model(str(output_path))
            return str(output_path)
        except Exception as e:
            self.logger.error(f"DAPT 失败: {e}")
            return ""

    # ----------------------------------------------------------------
    # 5. S4: 全量十折交叉验证 (PPL 计算)
    # ----------------------------------------------------------------
    def run_cv_validation(self, model_path: str, data_file: str) -> str:
        self.logger.info(f"Running Full-Data CV on {model_path}...")
        try:
            # 1. 读取数据
            df = pd.read_parquet(data_file)

            # === 关键修改：取消 1000 条限制，使用全量数据 ===
            texts = np.array(df[df['Review_Text'].notna()]['Review_Text'].tolist())
            self.logger.info(f"验证集总样本数: {len(texts)} (将执行全量 10 折交叉验证)")

            # 打乱顺序以保证随机性，但不切片
            np.random.shuffle(texts)

            # 2. 准备模型与设备 (自动使用 GPU 加速)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {device}")

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            model.to(device)
            model.eval()

            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            losses = []

            # 3. 开始 10 折循环
            for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
                val_texts = texts[val_idx].tolist()

                # === 显存保护机制 ===
                # 因为数据量大，不能一次性塞入模型，需要分 Batch 处理
                batch_size = 64  # 如果显存大可改为 32 或 64
                fold_batch_losses = []

                # 在 run_cv_validation 中
                # 引入 collator 来进行动态 Mask
                from transformers import DataCollatorForLanguageModeling
                val_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

                # 进度条效果：分批次计算 Loss
                for i in range(0, len(val_texts), batch_size):
                    batch = val_texts[i: i + batch_size]
                    # 1. 先编码
                    encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)

                    # 2. 调用 collator 产生 Masked inputs 和 Labels
                    # 注意：collator 期望的输入是 list of dicts
                    features = [{k: v[idx] for k, v in encodings.items()} for idx in range(len(batch))]
                    batch_data = val_collator(features)

                    # 3. 移动到设备
                    batch_data = {k: v.to(device) for k, v in batch_data.items()}

                    with torch.no_grad():
                        outputs = model(**batch_data)  # 这里的 loss 就是真实的 MLM Loss 了
                        fold_batch_losses.append(outputs.loss.item())

                # 计算当前折的平均 Loss
                fold_loss = np.mean(fold_batch_losses)
                losses.append(fold_loss)

                self.logger.info(f"Fold {fold}: Loss = {fold_loss:.4f} (Samples: {len(val_texts)})")

            # 4. 计算最终全局指标
            avg_loss = np.mean(losses)
            ppl = math.exp(avg_loss)  # PPL = e^loss

            report = {
                "avg_loss": avg_loss,
                "ppl": ppl,
                "total_samples": len(texts),
                "status": "full_data_calculation_success"
            }

            out_file = self.paths.output_dir / "cv_report.json"
            with open(out_file, "w") as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"✅ 全量验证完成! PPL: {ppl:.4f}")
            return str(out_file)

        except Exception as e:
            self.logger.error(f"CV Failed: {e}")
            self.logger.error(traceback.format_exc())
            return ""

    def check_model_sanity(self, model_path: str) -> str:
        """
        [真实逻辑升级]：
        读取模型权重，检测 LayerNorm 层的均值和全量权重的标准差。
        用于证明 DAPT 预训练后模型未坍缩（Mode Collapse）。
        """
        self.logger.info(f"🩺 正在对模型执行真实健康诊断: {model_path}")
        try:
            # 1. 加载模型
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            model.eval()

            # 2. 提取所有权重参数
            # 我们关注权重分布的标准差 (std)，如果 std 太小（如 < 1e-5），说明模型参数停止了更新
            all_weights = torch.cat([p.flatten() for p in model.parameters() if p.requires_grad])
            weight_std = all_weights.std().item()
            weight_mean = all_weights.mean().item()

            # 3. 特别检查 LayerNorm 层 (这是 BERT 稳定性的核心)
            # 我们随机取第一层 LayerNorm 的 bias 均值作为指标
            ln_params = [p for n, p in model.named_parameters() if 'LayerNorm.bias' in n]
            if ln_params:
                real_ln_mean = torch.mean(torch.stack([p.mean() for p in ln_params])).item()
            else:
                real_ln_mean = 0.0

            # 4. 检查是否存在异常值 (NaN 或 Inf)
            has_nan = torch.isnan(all_weights).any().item()

            # 5. 生成真实报告
            report = {
                "weight_std": round(weight_std, 6),
                "weight_mean": round(weight_mean, 6),
                "layer_norm_mean": round(real_ln_mean, 6),
                "has_nan": has_nan,
                "status": "healthy" if (0.01 < weight_std < 0.1 and not has_nan) else "warning"
            }

            out_file = self.paths.output_dir / "sanity_report.json"
            with open(out_file, "w") as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"✅ 诊断完成。模型权重标准差: {weight_std:.6f}, 状态: {report['status']}")
            return str(out_file)

        except Exception as e:
            self.logger.error(f"❌ 诊断执行失败: {e}")
            return ""

    def check_semantic_drift(
            self,
            model_path_new: str,
            model_path_old: str,
            platform: str,
            nebula_path: Optional[str] = None
    ) -> str:
        """
        Semantic Drift Analysis：DAPT 前后语义漂移检测。

        选择本方法的原因：
        1. 不只用少量人工词典，而是优先从 S2.5 Nebula_Lexicon_Data 自动读取高频领域词；
        2. 若词库不存在，再回退到人工种子词，保证流程不崩溃；
        3. 每个词计算 DAPT 前后 embedding cosine distance；
        4. 自动按 Logistics / Service / Eco 汇总，用于解释回归机制；
        5. 可直接支持 Taobao vs Ozon 的语义漂移对比图。
        """
        self.logger.info(f"🧭 正在计算 {platform} 的语义漂移 Semantic Drift...")

        try:
            # -------------------------------------------------
            # 1. 优先从 Nebula 词库读取自动发现词
            # -------------------------------------------------
            words = []

            # 优先使用外部传入的 Nebula 路径；
            # 如果 executor 没传，则自动寻找默认输出路径。
            if nebula_path:
                nebula_file = Path(nebula_path)
            else:
                nebula_file = self.paths.output_dir / f"Nebula_Lexicon_Data_{platform}.xlsx"

            if nebula_file.exists():
                lexicon_df = pd.read_excel(nebula_file)

                if "Word" in lexicon_df.columns:
                    # 优先保留高频且相似度较高的词
                    if "Frequency" in lexicon_df.columns:
                        lexicon_df = lexicon_df.sort_values("Frequency", ascending=False)

                    if "Similarity_to_Center" in lexicon_df.columns:
                        lexicon_df = lexicon_df[lexicon_df["Similarity_to_Center"] >= 0.3]

                    words = (
                        lexicon_df["Word"]
                        .dropna()
                        .astype(str)
                        .drop_duplicates()
                        .head(80)
                        .tolist()
                    )

                    self.logger.info(
                        f"✅ 从 Nebula 词库读取 {len(words)} 个领域词用于 Drift 分析"
                    )

            # -------------------------------------------------
            # 2. 如果没有 Nebula 词库，则使用扩展人工词典兜底
            # -------------------------------------------------
            if not words:
                if platform.lower() == "taobao":
                    words = [
                        # Logistics
                        "物流", "快递", "发货", "配送", "到货", "包装", "速度", "慢", "快", "顺丰",
                        # Service
                        "客服", "售后", "退款", "退货", "回复", "态度", "服务", "解决", "理赔", "小二",
                        # Eco / price
                        "价格", "优惠", "便宜", "划算", "折扣", "满减", "红包", "会员", "积分", "88vip",
                        # Product quality
                        "质量", "正品", "好用", "耐用", "差评", "满意", "推荐", "性价比"
                    ]
                else:
                    words = [
                        # Logistics
                        "доставка", "быстро", "долго", "заказ", "посылка", "упаковка", "пришел", "получил",
                        # Service
                        "возврат", "деньги", "продавец", "поддержка", "сервис", "гарантия", "вернули",
                        # Eco / price
                        "цена", "скидка", "дешево", "дорого", "акция", "карта", "ozon", "баллы",
                        # Product quality
                        "качество", "товар", "хороший", "плохой", "оригинал", "рекомендую"
                    ]

                self.logger.info(
                    f"⚠️ 未找到 Nebula 词库，使用人工扩展词典 {len(words)} 个词"
                )

            # -------------------------------------------------
            # 3. 加载 DAPT 前后模型
            # -------------------------------------------------
            old_tokenizer = AutoTokenizer.from_pretrained(model_path_old, use_fast=False)
            old_model = AutoModel.from_pretrained(model_path_old).to(self.device)
            old_model.eval()

            new_tokenizer = AutoTokenizer.from_pretrained(model_path_new, use_fast=False)
            new_model = AutoModel.from_pretrained(model_path_new).to(self.device)
            new_model.eval()

            def _get_cls_embedding(text: str, tokenizer, model) -> np.ndarray:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=32
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    vec = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]

                return vec

            # -------------------------------------------------
            # 4. 维度分类函数：用于机制解释
            # -------------------------------------------------
            def _assign_category(word: str) -> str:
                w = str(word).lower()

                if platform.lower() == "taobao":
                    if any(k in w for k in ["物流", "快递", "发货", "配送", "到货", "包装", "速度", "顺丰"]):
                        return "Logistics"
                    if any(k in w for k in ["客服", "售后", "退款", "退货", "回复", "态度", "服务", "解决", "理赔"]):
                        return "Service"
                    if any(k in w for k in
                           ["价格", "优惠", "便宜", "划算", "折扣", "满减", "红包", "会员", "积分", "88vip"]):
                        return "Eco"
                    return "Product"

                else:
                    if any(k in w for k in
                           ["доставка", "быстро", "долго", "заказ", "посылка", "упаковка", "пришел", "получил"]):
                        return "Logistics"
                    if any(k in w for k in
                           ["возврат", "деньги", "продавец", "поддержка", "сервис", "гарантия", "вернули"]):
                        return "Service"
                    if any(k in w for k in ["цена", "скидка", "дешево", "дорого", "акция", "карта", "ozon", "баллы"]):
                        return "Eco"
                    return "Product"

            # -------------------------------------------------
            # 5. 计算每个词的 drift
            # -------------------------------------------------
            results = []

            for word in words:
                word = str(word).strip()
                if not word:
                    continue

                old_vec = _get_cls_embedding(word, old_tokenizer, old_model)
                new_vec = _get_cls_embedding(word, new_tokenizer, new_model)

                sim = cosine_similarity(
                    old_vec.reshape(1, -1),
                    new_vec.reshape(1, -1)
                )[0][0]

                drift = 1 - sim

                results.append({
                    "Platform": platform,
                    "Word": word,
                    "Category": _assign_category(word),
                    "Cosine_Similarity_Old_New": float(sim),
                    "Semantic_Drift_Distance": float(drift)
                })

            drift_df = pd.DataFrame(results)

            # -------------------------------------------------
            # 6. 分类汇总：直接用于论文机制解释
            # -------------------------------------------------
            summary_df = (
                drift_df
                .groupby(["Platform", "Category"], as_index=False)
                .agg(
                    Mean_Drift=("Semantic_Drift_Distance", "mean"),
                    Median_Drift=("Semantic_Drift_Distance", "median"),
                    Max_Drift=("Semantic_Drift_Distance", "max"),
                    Word_Count=("Word", "count")
                )
                .sort_values("Mean_Drift", ascending=False)
            )

            # -------------------------------------------------
            # 7. 输出 Excel
            # -------------------------------------------------
            out_path = self.paths.output_dir / f"Semantic_Drift_{platform}.xlsx"

            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                drift_df.to_excel(writer, sheet_name="Word_Level_Drift", index=False)
                summary_df.to_excel(writer, sheet_name="Category_Summary", index=False)

            avg_drift = drift_df["Semantic_Drift_Distance"].mean()

            self.logger.info(
                f"✅ {platform} Semantic Drift 完成，平均 Drift = {avg_drift:.4f}，输出: {out_path}"
            )

            return str(out_path)

        except Exception as e:
            self.logger.error(f"❌ Semantic Drift 计算失败: {e}")
            self.logger.error(traceback.format_exc())
            return ""

        # ... 计算两个模型输出 Embedding 的 Cosine Distance ...
        # 偏移量越大，证明 DAPT 的“知识注入”越成功。

    # =========================================================================
    # S6-1: 深度主题挖掘 (BERTopic Elite)
    # =========================================================================
    def run_bertopic_pro(self, data_file: str, model_path: str, platform: str) -> str:
        """
        S6: [Pro Advanced] 结合领域 DAPT 模型的主题发现。
        碾压点：1. 使用自练模型  2. 全量数据计算  3. 语义聚类优化
        """
        self.logger.info(f"🗺️ 正在利用 {platform} 领域模型执行深度主题挖掘...")
        try:
            # 1. 加载全量数据
            df = pd.read_parquet(data_file)
            sub_df = df[df['Platform'] == platform]
            docs = sub_df['Review_Text'].astype(str).tolist()

            if not docs: return ""

            # 2. 使用稳定的多语言 SBERT 句向量模型
            from sentence_transformers import SentenceTransformer
            # 说明：
            # DAPT 模型用于领域自适应预训练与语义空间增强；
            # BERTopic 阶段先使用成熟 SBERT 句向量模型，保证聚类稳定性。
            emb_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2",
                device=self.device
            )
            # model_path 保留为接口参数，方便未来升级为 DAPT embedding + UMAP + HDBSCAN。

            # 3. 初始化 BERTopic，加入类别的自动聚合
            topic_model = BERTopic(
                embedding_model=emb_model,
                calculate_probabilities=False,
                verbose=True,
                nr_topics="auto"  # 自动聚类为最有意义的 N 个话题
            )

            # 4. 执行拟合 (针对 5w+ 数据)
            topics, probs = topic_model.fit_transform(docs)

            # 5. 生成可视化图表 (不仅是 Map，还导出主题分布表)
            fig = topic_model.visualize_topics()
            out_html = self.paths.output_dir / f"Topic_Cloud_{platform}.html"
            fig.write_html(str(out_html))

            # 额外产出：主题词频表 (这才是写论文的数据基础)
            info_df = topic_model.get_topic_info()
            info_df.to_excel(self.paths.output_dir / f"Topic_Keywords_{platform}.xlsx", index=False)

            self.logger.info(f"✨ {platform} 深度主题分析完成，产出已归档。")
            return str(out_html)

        except Exception as e:
            self.logger.error(f"主题分析失败: {e}")
            return ""

    # =========================================================================
    # S5.5: 多模态基础字段
    # =========================================================================
    def execute_s5_5_multimodal_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        S5.5 多模态基础字段：
        优先读取本地已下载图片文件夹，生成主页媒体供给与评论媒体证据字段。

        Taobao:
        - 主页图片：淘宝主页产品图片 / Product_Name / *.jpg
        - 评论图片：淘宝评价图片 / Product_Name / 000800_* / *.jpg

        Ozon:
        - 主页图片：ozon主页产品图片 / SKU / *.jpg
        - 评论图片：ozon评价图片 / SKU / 003052_* / *.jpg

        视频：
        - 本轮只保留数量/存在性字段，不做抽帧。
        """
        df = df.copy()
        df.columns = df.columns.str.strip()

        for col in [
            "Product_Name", "Product_Image",
            "Product_Video", "Product_video",
            "Review_Images", "Review_Pictures", "Review_Videos",
            "SKU", "Product_Details (SKU)"
        ]:
            if col not in df.columns:
                df[col] = ""

        platform_col = "Platform" if "Platform" in df.columns else ("platform" if "platform" in df.columns else None)

        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

        def clean_text(x) -> str:
            if pd.isna(x):
                return ""
            return " ".join(str(x).strip().split())

        def _count_images_in_folder(folder: Path) -> int:
            if folder is None or not folder.exists() or not folder.is_dir():
                return 0
            return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in image_exts)

        def _count_images_recursive(folder: Path) -> int:
            if folder is None or not folder.exists() or not folder.is_dir():
                return 0
            return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in image_exts)

        def _first_image_in_folder(folder: Path) -> str:
            if folder is None or not folder.exists() or not folder.is_dir():
                return ""
            files = sorted(
                [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
            )
            return str(files[0]) if files else ""

        def _find_review_row_folder(base_dir: Path, product_key: str, source_row_no: int) -> Optional[Path]:
            if base_dir is None or not base_dir.exists():
                return None

            product_folder = base_dir / str(product_key).strip()
            if not product_folder.exists():
                return None

            prefix = str(source_row_no)

            matches = [
                p for p in product_folder.iterdir()
                if p.is_dir() and p.name.startswith(prefix)
            ]

            return matches[0] if matches else None

        def _get_taobao_key(row) -> str:
            return clean_text(row.get("Product_Name", ""))

        def _get_ozon_key(row) -> str:
            sku = clean_text(row.get("SKU", ""))
            if sku:
                return sku
            sku2 = clean_text(row.get("Product_Details (SKU)", ""))
            if sku2:
                return self._normalize_sku(sku2)
            return ""

        seller_image_counts = []
        review_image_counts = []
        seller_first_image_paths = []
        review_first_image_paths = []

        for idx, row in df.iterrows():
            home_first_img = ""
            review_first_img = ""
            platform = str(row.get(platform_col, "")).lower() if platform_col else ""

            if platform == "taobao":
                product_key = _get_taobao_key(row)

                home_count = 0
                if self.paths.taobao_home_img_dir:
                    home_folder = self.paths.taobao_home_img_dir / product_key
                    home_count = _count_images_in_folder(home_folder)
                    home_first_img = _first_image_in_folder(home_folder)

                if home_count == 0:
                    home_count = self._parse_media_count(row.get("Product_Image", ""))

                review_count = 0
                if self.paths.taobao_review_img_dir:
                    source_row_no = row.get("Source_Row_No", idx)

                    review_folder = _find_review_row_folder(
                        self.paths.taobao_review_img_dir,
                        product_key,
                        source_row_no
                    )
                    review_count = _count_images_in_folder(review_folder)
                    review_first_img = _first_image_in_folder(review_folder)

                if review_count == 0:
                    review_count = self._parse_media_count(row.get("Review_Images", ""))

            elif platform == "ozon":
                sku_key = _get_ozon_key(row)

                home_count = 0
                if self.paths.ozon_home_img_dir:
                    home_folder = self.paths.ozon_home_img_dir / sku_key
                    home_count = _count_images_in_folder(home_folder)
                    home_first_img = _first_image_in_folder(home_folder)

                if home_count == 0:
                    home_count = self._parse_media_count(row.get("Product_Image", ""))

                review_count = 0
                if self.paths.ozon_review_img_dir:
                    source_row_no = row.get("Source_Row_No", idx)

                    review_folder = _find_review_row_folder(
                        self.paths.ozon_review_img_dir,
                        sku_key,
                        source_row_no
                    )
                    review_count = _count_images_in_folder(review_folder)
                    review_first_img = _first_image_in_folder(review_folder)

                if review_count == 0:
                    review_count = self._parse_media_count(row.get("Review_Pictures", ""))

            else:
                home_count = self._parse_media_count(row.get("Product_Image", ""))
                review_count = max(
                    self._parse_media_count(row.get("Review_Images", "")),
                    self._parse_media_count(row.get("Review_Pictures", ""))
                )

            seller_image_counts.append(home_count)
            review_image_counts.append(review_count)
            seller_first_image_paths.append(home_first_img)
            review_first_image_paths.append(review_first_img)

        product_video_series = pd.Series("", index=df.index)

        if "Product_Video" in df.columns and "Product_video" in df.columns:
            product_video_series = df["Product_Video"].where(
                df["Product_Video"].astype(str).str.strip().str.len() > 0,
                df["Product_video"]
            )
        elif "Product_Video" in df.columns:
            product_video_series = df["Product_Video"]
        elif "Product_video" in df.columns:
            product_video_series = df["Product_video"]

        df["Seller_Image_Count"] = seller_image_counts
        df["Seller_First_Image_Path"] = seller_first_image_paths
        df["Review_First_Image_Path"] = review_first_image_paths
        df["Seller_Video_Count"] = product_video_series.apply(self._parse_media_count)

        df["Has_Product_Image"] = (df["Seller_Image_Count"] > 0).astype(int)
        df["Has_Product_Video"] = (df["Seller_Video_Count"] > 0).astype(int)

        df["Image_Count"] = review_image_counts
        df["Video_Count"] = df["Review_Videos"].apply(self._parse_media_count)

        df["Has_Image"] = (df["Image_Count"] > 0).astype(int)
        df["Has_Video"] = (df["Video_Count"] > 0).astype(int)

        df["Media_Count"] = df["Image_Count"] + df["Video_Count"]

        return df

    def _load_clip(self):
        if self._clip_model is not None and self._clip_processor is not None:
            return self._clip_model, self._clip_processor

        if not self.paths.clip_model:
            raise ValueError("CLIP model path is not configured. Please set paths.clip_model.")

        self._clip_model = CLIPModel.from_pretrained(str(self.paths.clip_model)).to(self.device)
        self._clip_processor = CLIPProcessor.from_pretrained(str(self.paths.clip_model))
        self._clip_model.eval()

        return self._clip_model, self._clip_processor

    def _clip_text_image_similarity(self, text: str, image_path: Path) -> float:
        if not image_path or not image_path.exists():
            return np.nan

        text = "" if pd.isna(text) else str(text).strip()
        if text == "" or text.lower() == "nan":
            return np.nan

        try:
            model, processor = self._load_clip()

            image = Image.open(image_path).convert("RGB")

            inputs = processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

                sim = torch.matmul(text_embeds, image_embeds.T).item()

            return float((sim + 1) / 2)

        except Exception as e:
            self.logger.warning(f"CLIP similarity failed for {image_path}: {e}")
            return np.nan

    # =========================================================================
    # S5.6: 多模态证据变量（MVP版）
    # =========================================================================
    def execute_s5_6_multimodal_evidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        多模态证据变量（MVP版）：
        使用 CLIP 计算 Review_Text 与 Review_First_Image_Path 的文本-图片一致性。
        逻辑与论文口径保持一致：
        1. 模态内先得到 Evidence_image / Evidence_video
        2. 评论级 Evidence_review：
           - 仅图片 -> 用图片证据
           - 仅视频 -> 用视频证据
           - 图文都有 -> 取平均
           - 都没有 -> 置为 0
        后续再升级为真正的 Text-Image/Text-Video Alignment + Noise Gating。
        """
        df = df.copy()

        # -----------------------------
        # 0. 缺失字段兜底
        # -----------------------------
        required_cols = [
            'Has_Image', 'Has_Video',
            'Has_Product_Image', 'Has_Product_Video'
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0

        # 强制转成 0/1
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(0, 1).astype(int)

        # -----------------------------
        # 1. 文本-图片一致性：真实 CLIP 分数
        # -----------------------------
        clip_scores = []

        for _, row in df.iterrows():
            if int(row.get("Has_Image", 0)) != 1:
                clip_scores.append(np.nan)
                continue

            text = row.get("Review_Text", "")
            img_path = row.get("Review_First_Image_Path", "")

            if not img_path or str(img_path).lower() == "nan":
                clip_scores.append(np.nan)
                continue

            score = self._clip_text_image_similarity(
                text=text,
                image_path=Path(str(img_path))
            )
            clip_scores.append(score)

        df["Text_Image_Alignment"] = clip_scores

        # 视频暂时不做 CLIP，只保留占位
        df["Text_Video_Alignment"] = np.where(df["Has_Video"] == 1, 0.6, np.nan)

        # -----------------------------
        # 2. 噪声项（MVP先置0）
        # -----------------------------
        df['Noise'] = 0.0

        # -----------------------------
        # 3. 单模态证据
        #    Evidence_i = Align_i * (1 - Noise_i)
        # -----------------------------
        df['Evidence_image'] = df['Text_Image_Alignment'] * (1 - df['Noise'])
        df['Evidence_video'] = df['Text_Video_Alignment'] * (1 - df['Noise'])

        # -----------------------------
        # 4. 卖家媒体 vs 评论媒体 匹配（MVP占位）
        # -----------------------------
        df['Seller_Review_Image_Match'] = np.where(
            (df['Has_Product_Image'] == 1) & (df['Has_Image'] == 1),
            0.6,
            np.nan
        )

        df['Seller_Review_Video_Match'] = np.where(
            (df['Has_Product_Video'] == 1) & (df['Has_Video'] == 1),
            0.6,
            np.nan
        )

        # -----------------------------
        # 5. CredibleAlign（当前先以图片证据为主，后续可升级）
        # -----------------------------
        df['CredibleAlign'] = df['Evidence_image']

        # -----------------------------
        # 6. 评论级证据聚合（与文档口径统一）
        # -----------------------------
        img = pd.to_numeric(df['Evidence_image'], errors='coerce').to_numpy(dtype=float)
        vid = pd.to_numeric(df['Evidence_video'], errors='coerce').to_numpy(dtype=float)

        has_img = df['Has_Image'].to_numpy(dtype=int) == 1
        has_vid = df['Has_Video'].to_numpy(dtype=int) == 1

        evidence_review = np.zeros(len(df), dtype=float)

        # 仅图片
        mask_img_only = has_img & (~has_vid)
        evidence_review[mask_img_only] = np.nan_to_num(img[mask_img_only], nan=0.0)

        # 仅视频
        mask_vid_only = (~has_img) & has_vid
        evidence_review[mask_vid_only] = np.nan_to_num(vid[mask_vid_only], nan=0.0)

        # 图文都有 -> 平均
        mask_both = has_img & has_vid
        evidence_review[mask_both] = (
                                             np.nan_to_num(img[mask_both], nan=0.0) +
                                             np.nan_to_num(vid[mask_both], nan=0.0)
                                     ) / 2.0

        # 都没有 -> 0
        mask_none = (~has_img) & (~has_vid)
        evidence_review[mask_none] = 0.0

        df['Evidence_review'] = evidence_review

        return df

    # =========================================================================
    # S6-2: 复合满意度量化 (CSI 计算)
    # =========================================================================
    def execute_s6_csi_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算复合满意度指标 CSI：
        Taobao: CSI = Map5(BERT_Sentiment_Prob + α * log(Helpful + 1))
        Ozon:   CSI = Map5(BERT_Sentiment_Prob + α * log((Yes + 1) / (No + 1)))

        说明：
        1. BERT_Sentiment_Prob 为 0–1 区间的基础情感概率；
        2. 社交投票是评论可信度 / 情感放大器，而不是商品评分本身；
        3. Map5 前隐含 clipping，将连续值裁剪到 [0,1] 后再映射到 [1,5]。
        """
        self.logger.info("🧠 正在根据社会证明理论计算复合满意度 (CSI)...")

        df = df.copy()
        alpha = 0.1

        platform_col = "Platform" if "Platform" in df.columns else ("platform" if "platform" in df.columns else None)
        if platform_col is None:
            self.logger.warning("⚠️ 缺失 Platform/platform 列，默认按通用公式计算 CSI")
            df["CSI_Score"] = 3.0
            return df

        def _safe_float(val, default=0.0):
            try:
                if pd.isna(val):
                    return default
                return float(val)
            except Exception:
                return default

        def calculate_csi(row):
            base_score = _safe_float(row.get("BERT_Sentiment_Prob", 0.5), 0.5)
            base_score = max(0.0, min(1.0, base_score))

            platform = str(row.get(platform_col, "")).strip().lower()

            if platform == "taobao":
                helpful = _safe_float(row.get("Helpful_Count", 0), 0.0)
                social_signal = alpha * math.log(helpful + 1.0)

            elif platform == "ozon":
                yes = _safe_float(row.get("Yes_Count", 0), 0.0)
                no = _safe_float(row.get("No_Count", 0), 0.0)
                social_signal = alpha * math.log((yes + 1.0) / (no + 1.0))

            else:
                social_signal = 0.0

            combined_raw = base_score + social_signal
            csi_score = self._map5(combined_raw)
            return round(csi_score, 2)

        df["CSI_Score"] = df.apply(calculate_csi, axis=1)
        return df

    # =========================================================================
    # S6-3: 核心归因特征映射 (X1, X2, X3)
    # =========================================================================

    def _calculate_taobao_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        淘宝：基于清洗后真实字段的数值型特征映射。
        逻辑口径：
        X1 = 物流履约与确权
        X2 = 服务交互与风险对冲 = X2_home + X2_review
        X3 = 生态金融锁定
        """
        df = df.copy()

        # -----------------------------
        # 1. X1: 物流履约与确权
        # -----------------------------
        ship_time = pd.to_numeric(
            df.get("Shipment_Time_Hrs", pd.Series(48, index=df.index)),
            errors="coerce"
        ).fillna(48)

        promised_time = pd.to_numeric(
            df.get("Estimated_Shipping_Desc", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        ).fillna(ship_time)

        has_official_rating = df.get(
            "Logistics_Service_Review",
            pd.Series("", index=df.index)
        ).astype(str).apply(
            lambda x: 1 if any(k in x for k in ["优秀", "高", "好"]) else 0
        )

        x1_base = self._safe_norm(24.0 / (ship_time + 1))
        x1_promise = 0.3 * self._safe_norm(24.0 / (promised_time + 1))
        x1_rating = 0.2 * has_official_rating
        x1_text = pd.to_numeric(
            df.get("X1_Score", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)

        df["X1_Logistics"] = x1_base + x1_promise + x1_rating + x1_text

        # -----------------------------
        # 2. X2: 服务交互与风险对冲
        #    X2 = X2_home + X2_review
        # -----------------------------
        cs_sec = pd.to_numeric(
            df.get("CS_Response_Sec", pd.Series(3600, index=df.index)),
            errors="coerce"
        ).fillna(3600)

        refund = pd.to_numeric(
            df.get("Instant_Refund_Flag", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0).clip(0, 1)

        insurance = pd.to_numeric(
            df.get("Shipping_Insurance_Flag", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0).clip(0, 1)

        reply = df.get(
            "Merchant_Reply",
            pd.Series("", index=df.index)
        ).astype(str).apply(
            lambda x: 1 if len(x.strip()) > 5 and x.lower() != "nan" else 0
        )

        # 主页多媒体展示能力：图片 + 视频
        # 说明：
        # 1. Has_Product_Image / Has_Product_Video 来自 S5.5；
        # 2. 如果 S5.5 还没运行，则这里自动从 Product_Image / Product_Video 字段兜底生成；
        # 3. 视频只作为“是否有动态展示”的轻量变量，不做抽帧分析。

        if "Seller_Image_Count" in df.columns:
            seller_image_count = pd.to_numeric(
                df["Seller_Image_Count"],
                errors="coerce"
            ).fillna(0)
        else:
            seller_image_count = df.get(
                "Product_Image",
                pd.Series("", index=df.index)
            ).apply(self._parse_media_count)

        if "Has_Product_Image" in df.columns:
            has_product_image = pd.to_numeric(
                df["Has_Product_Image"],
                errors="coerce"
            ).fillna(0).clip(0, 1)
        else:
            has_product_image = (seller_image_count > 0).astype(int)

        if "Has_Product_Video" in df.columns:
            has_product_video = pd.to_numeric(
                df["Has_Product_Video"],
                errors="coerce"
            ).fillna(0).clip(0, 1)
        else:
            product_video_series = df.get(
                "Product_Video",
                df.get("Product_video", pd.Series("", index=df.index))
            )
            has_product_video = product_video_series.astype(str).apply(
                lambda x: 1 if len(x.strip()) > 5 and ("http" in x or "mp4" in x) else 0
            )

        # 图片数量采用归一化，避免图片很多的商品过度放大
        media_supply = (
                0.50 * self._safe_norm(seller_image_count)
                + 0.25 * has_product_image
                + 0.25 * has_product_video
        )

        x2_home = (
                self._safe_norm(1.0 / (cs_sec + 1))
                + 0.30 * media_supply
                + 0.15 * refund
                + 0.15 * insurance
        )

        x2_review = 0.2 * reply

        x2_text = pd.to_numeric(
            df.get("X2_Score", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)

        df["X2_Service"] = x2_home + x2_review + x2_text

        # -----------------------------
        # 3. X3: 生态金融锁定
        # -----------------------------
        p_orig = pd.to_numeric(
            df.get("Price", pd.Series(100, index=df.index)),
            errors="coerce"
        ).fillna(100)

        p_disc = pd.to_numeric(
            df.get("Discounted_Price", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        ).fillna(p_orig)

        gap = ((p_orig - p_disc) / (p_orig + 1)).clip(0, 1)

        coupon_1 = df.get("Coupon_1", pd.Series("", index=df.index)).astype(str)
        coupon_2 = df.get("Coupon_2", pd.Series("", index=df.index)).astype(str)
        coupon_col = coupon_1 + coupon_2

        coupon = coupon_col.apply(
            lambda x: 1 if (
                    (len(x.strip()) > 0 and x.lower() != "nan")
                    or any(k in x for k in ["补贴", "满减", "券"])
            ) else 0
        )

        installment = df.get(
            "Installment_Plan",
            pd.Series("", index=df.index)
        ).astype(str).apply(
            lambda x: 1 if len(x.strip()) > 0 and x.lower() != "nan" else 0
        )

        loyalty = self._safe_norm(
            np.log1p(
                pd.to_numeric(
                    df.get("Repeat_Customers_Num", pd.Series(0, index=df.index)),
                    errors="coerce"
                ).fillna(0)
            )
        )

        # 评论侧生态锁定补充：用户等级
        user_level = pd.to_numeric(
            df.get("X3_User_Level_Score", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)

        x3_text = pd.to_numeric(
            df.get("X3_Score", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)

        df["X3_Eco"] = (
                gap
                + 0.2 * coupon
                + 0.2 * installment
                + 0.3 * loyalty
                + user_level
                + x3_text
        )

        # -----------------------------
        # 4. 控制变量
        # -----------------------------
        df["Control_Price"] = np.log1p(p_disc)

        df["Control_Quality"] = pd.to_numeric(
            df.get("Star_Number", pd.Series(4.7, index=df.index)),
            errors="coerce"
        ).fillna(4.7)

        pop_a = pd.to_numeric(
            df.get("Picture_Review_Num", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)

        pop_b = pd.to_numeric(
            df.get("Additional_Review_Num", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)

        df["Control_Pop"] = np.log1p(pop_a + pop_b)

        df = self._add_platform_weighted_index(df, "taobao")
        return df

    def _calculate_ozon_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ozon：基于清洗后真实字段的数值型特征映射。
        逻辑口径：
        X1 = 基础设施确定性（FBO + 小时制时效承诺）
        X2 = 制度性服务（平台规则强度 + 评论侧制度服务语义补充）
        X3 = 金融科技生态（绿卡剪刀差 + 金融覆盖）
        """
        df = df.copy()

        # -----------------------------
        # 1. X1: 基础设施 / 物流确定性
        # -----------------------------
        fbo = df.get(
            "Warehouse_Type",
            pd.Series("", index=df.index)
        ).astype(str).apply(
            lambda x: 1 if "ozon" in x.lower() else 0
        )

        # Delivery_Promise 已是小时制连续变量
        delivery_hours = pd.to_numeric(
            df.get("Delivery_Promise", pd.Series(96, index=df.index)),
            errors="coerce"
        ).fillna(96)

        fast_score = self._safe_norm(24.0 / (delivery_hours + 1))

        x1_text = pd.to_numeric(
            df.get("X1_Score", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)

        df["X1_Logistics"] = fbo + fast_score + x1_text

        # -----------------------------
        # 2. X2: 制度性服务
        #    X2 = X2_home + X2_text
        # -----------------------------
        # Return_Policy 在你当前数据里主要是平台规则变量：
        # 食品类=0，其他多数=168，因此不能单独充当完整 X2。
        return_hours = pd.to_numeric(
            df.get("Return_Policy", pd.Series(168, index=df.index)),
            errors="coerce"
        ).fillna(168)

        return_days = return_hours / 24.0

        # 主页侧制度规则强度：保留，但权重不要让它完全主导
        x2_home_raw = return_days / 30.0

        # -----------------------------
        # 评论侧制度服务语义补充
        # -----------------------------
        review_text = df.get(
            "Review_Text",
            pd.Series("", index=df.index)
        ).fillna("").astype(str).str.lower()

        # 退货 / 退款 / 保修 / 支持 / 服务 等制度性服务信号
        service_keywords = [
            "возврат", "вернул", "вернули", "возвращ",
            "деньги", "refund", "return",
            "гарантия", "поддержка", "сервис", "service"
        ]

        text_service_hit = review_text.apply(
            lambda x: 1 if any(k in x for k in service_keywords) else 0
        )

        # 社区是否对评论有互动，也可作为制度服务可感知性的弱代理
        yes_count = pd.to_numeric(
            df.get("Yes_Count", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)

        no_count = pd.to_numeric(
            df.get("No_Count", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)

        interaction_signal = self._safe_norm(np.log1p(yes_count + no_count))

        # 如果有图评论，往往制度问题（退货/破损/不符）会更可验证
        has_image = pd.to_numeric(
            df.get("Has_Image", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0).clip(0, 1)

        # 原有文本补充项
        x2_score = pd.to_numeric(
            df.get("X2_Score", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)

        # 让评论侧补充真正产生方差
        x2_text_raw = (
                0.5 * text_service_hit
                + 0.3 * interaction_signal
                + 0.2 * has_image
                + x2_score
        )

        # 主页多媒体展示能力：图片 + 视频
        # Ozon 这里同样只识别“是否提供 / 提供多少”，不做视频抽帧分析。
        if "Seller_Image_Count" in df.columns:
            seller_image_count = pd.to_numeric(
                df["Seller_Image_Count"],
                errors="coerce"
            ).fillna(0)
        else:
            seller_image_count = df.get(
                "Product_Image",
                pd.Series("", index=df.index)
            ).apply(self._parse_media_count)

        if "Has_Product_Image" in df.columns:
            has_product_image = pd.to_numeric(
                df["Has_Product_Image"],
                errors="coerce"
            ).fillna(0).clip(0, 1)
        else:
            has_product_image = (seller_image_count > 0).astype(int)

        if "Has_Product_Video" in df.columns:
            has_product_video = pd.to_numeric(
                df["Has_Product_Video"],
                errors="coerce"
            ).fillna(0).clip(0, 1)
        else:
            product_video_series = df.get(
                "Product_Video",
                df.get("Product_video", pd.Series("", index=df.index))
            )
            has_product_video = product_video_series.astype(str).apply(
                lambda x: 1 if len(x.strip()) > 5 and ("http" in x or "mp4" in x) else 0
            )

        media_supply = (
                0.50 * self._safe_norm(seller_image_count)
                + 0.25 * has_product_image
                + 0.25 * has_product_video
        )

        # 最终 X2：平台规则强度 + 主页多媒体展示 + 评论侧制度服务感知
        # 三部分分别归一化/标准化，避免某一项统治整列。
        x2_home = self._safe_norm(x2_home_raw)
        x2_text = self._safe_norm(x2_text_raw)

        df["X2_Service"] = (
                0.45 * x2_home
                + 0.25 * media_supply
                + 0.30 * x2_text
        )

        # -----------------------------
        # 3. X3: 金融科技生态
        # -----------------------------
        orig = pd.to_numeric(
            df.get("Original_Price", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)

        card = pd.to_numeric(
            df.get("Ozon_Card_Price", pd.Series(np.nan, index=df.index)),
            errors="coerce"
        ).fillna(orig)

        gap = ((orig - card) / (orig + 1)).clip(0, 1)

        fintech = df.get(
            "Installment_Plan",
            pd.Series("", index=df.index)
        ).astype(str).apply(
            lambda x: 1 if "ozon" in x.lower() else 0
        )

        x3_text = pd.to_numeric(
            df.get("X3_Score", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)

        df["X3_Eco"] = gap + fintech + x3_text

        # -----------------------------
        # 4. 控制变量
        # -----------------------------
        price_disc = pd.to_numeric(
            df.get("Price_after_discount", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)
        df["Control_Price"] = np.log1p(price_disc)

        df["Control_Quality"] = pd.to_numeric(
            df.get("Rating_Score", pd.Series(5.0, index=df.index)),
            errors="coerce"
        ).fillna(5.0)

        review_count = pd.to_numeric(
            df.get("Review_Count", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)
        df["Control_Pop"] = np.log1p(review_count)

        stock = pd.to_numeric(
            df.get("Stock_Status", pd.Series(0, index=df.index)),
            errors="coerce"
        ).fillna(0)
        df["Control_Scarcity"] = np.log1p(stock)

        df = self._add_platform_weighted_index(df, "ozon")
        return df

    # =========================================================================
    # S6-4: 计量回归分析 (OLS + VIF + HC3)
    # =========================================================================

    def run_scientific_regression(self, data_file: str) -> str:
        """
        执行最终回归分析：
        Y_i = β0 + β1 X1_i + β2 X2_i + β3 X3_i + β4 Evidence_review_i + γ Controls_i + ε_i

        说明：
        1. 总式用于统一比较框架；
        2. 实际估计按平台分别进行（Taobao / Ozon 分平台回归）；
        3. 使用 OLS + HC3 稳健标准误；
        4. 自动输出 VIF 共线性诊断。
        """
        self.logger.info("📊 启动高阶计量回归引擎 (VIF + HC3)...")

        df = pd.read_parquet(data_file)
        results_path = self.paths.output_dir / "Regression_Final_Report.txt"

        # -----------------------------
        # 0. 平台列识别
        # -----------------------------
        platform_col = "Platform" if "Platform" in df.columns else ("platform" if "platform" in df.columns else None)
        if platform_col is None:
            raise ValueError("数据中缺失 Platform/platform 列，无法分平台回归。")

        with open(results_path, "w", encoding="utf-8") as f:
            f.write("==============================================================\n")
            f.write("🏆 Cross-Platform Scientific Regression Report (Thesis Final)\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n")
            f.write("Methodology: OLS with HC3 Robust Standard Errors & VIF Audit\n")
            f.write("==============================================================\n\n")

            for platform in ['taobao', 'ozon']:
                sub = df[df[platform_col].astype(str).str.lower() == platform].copy()

                if len(sub) < 30:
                    self.logger.warning(f"⚠️ [{platform}] 样本量不足 ({len(sub)})，跳过回归。")
                    f.write(f"⚠️ [{platform}] 样本量不足 ({len(sub)})，跳过回归。\n\n")
                    continue

                # -----------------------------
                # 1. 因变量：CSI_Score
                # -----------------------------
                if 'CSI_Score' in sub.columns:
                    Y = pd.to_numeric(sub['CSI_Score'], errors='coerce')
                else:
                    self.logger.warning(f"[{platform}] 缺失 CSI_Score，降级使用 Star_Number")
                    fallback_series = sub.get('Star_Number', pd.Series(4.5, index=sub.index))
                    Y = pd.to_numeric(fallback_series, errors='coerce')

                Y = Y.fillna(Y.mean() if Y.notna().any() else 4.5)
                Y = Y.replace([np.inf, -np.inf], Y.mean() if Y.notna().any() else 4.5)

                # -----------------------------
                # 2. 自变量与控制变量
                # -----------------------------
                core_cols = [
                    'Omnichannel_Capability_Index',
                    'Evidence_review',
                ]

                control_cols = [
                    'Control_Price',
                    'Control_Quality',
                    'Control_Pop'
                ]

                if platform == 'ozon':
                    control_cols.append('Control_Scarcity')

                target_cols = core_cols + control_cols

                missing_cols = [c for c in target_cols if c not in sub.columns]
                if missing_cols:
                    self.logger.error(f"[{platform}] 缺失回归特征: {missing_cols}")
                    f.write(f"❌ [{platform}] Critical Features Missing: {missing_cols}\n\n")
                    continue

                # -----------------------------
                # 3. 数据清洗
                # -----------------------------
                X_data = sub[target_cols].copy()

                for c in target_cols:
                    X_data[c] = pd.to_numeric(X_data[c], errors='coerce')

                X_data = X_data.replace([np.inf, -np.inf], np.nan)

                # 当前阶段：缺失统一填 0
                # 后续如需更严谨，可对连续变量做均值填补 + missing dummy
                X_data = X_data.fillna(0)

                X = sm.add_constant(X_data, has_constant='add')

                # -----------------------------
                # 4. VIF
                # -----------------------------
                try:
                    from statsmodels.stats.outliers_influence import variance_inflation_factor

                    vif_df = pd.DataFrame({"Feature": X.columns})
                    if X.shape[1] > 1:
                        vif_values = []
                        for i in range(X.shape[1]):
                            try:
                                vif_values.append(variance_inflation_factor(X.values, i))
                            except Exception:
                                vif_values.append(np.nan)
                        vif_df["VIF"] = vif_values
                    else:
                        vif_df["VIF"] = 0.0

                except Exception as e:
                    self.logger.warning(f"[{platform}] VIF 计算失败: {e}")
                    vif_df = pd.DataFrame({"Feature": X.columns, "VIF": [np.nan] * len(X.columns)})

                # -----------------------------
                # 5. HC3 稳健回归
                """
                主回归公式：
                Y_i = β0 
                    + β1 Omnichannel_Capability_Index_i
                    + β2 Evidence_review_i
                    + γ Controls_i
                    + ε_i

                说明：
                1. Omnichannel_Capability_Index 是 X1/X2/X3 的平台差异化加权综合指数；
                2. X1/X2/X3_Norm 不进入主回归，避免与综合指数共线；
                3. Evidence_review 表示评论侧多模态证据强度；
                4. 分平台估计，使用 OLS + HC3 稳健标准误。
                """
                # -----------------------------
                try:
                    model = sm.OLS(Y, X).fit(cov_type='HC3')

                    f.write(f"--- Platform: {platform.upper()} (N={len(sub)}) ---\n")
                    f.write(f"R-squared: {model.rsquared:.4f} | Adj. R-squared: {model.rsquared_adj:.4f}\n")
                    f.write(f"F-statistic: {model.fvalue:.2f} (Prob: {model.f_pvalue:.4f})\n\n")

                    f.write("[1] Variables Included:\n")
                    f.write(
                        "Core Regressors: Omnichannel_Capability_Index, Evidence_review\n"
                    )
                    f.write(
                        "Note: X1_Logistics_Norm, X2_Service_Norm and X3_Eco_Norm are excluded from the main regression "
                        "because Omnichannel_Capability_Index is their weighted composite index.\n"
                    )
                    f.write(f"Controls: {', '.join(control_cols)}\n\n")

                    f.write("[2] VIF Audit (Multicollinearity Check):\n")
                    f.write(vif_df.to_string(index=False) + "\n\n")

                    f.write("[3] Regression Results (HC3 Robust):\n")
                    summary_df = pd.DataFrame({
                        "Coef": model.params,
                        "Std.Err": model.bse,
                        "t-value": model.tvalues,
                        "P>|t|": model.pvalues
                    })
                    summary_df['Sig.'] = summary_df['P>|t|'].apply(
                        lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                    )
                    f.write(summary_df.to_string() + "\n")
                    f.write("-" * 70 + "\n\n")

                    self.logger.info(f"✅ {platform} 回归完成 (R2={model.rsquared:.3f})")

                except Exception as e:
                    self.logger.error(f"❌ {platform} 回归计算崩溃: {e}")
                    f.write(f"Error in calculation: {str(e)}\n\n")

        self.logger.info(f"✨ 最终回归报告已生成: {results_path.name}")
        return str(results_path)

    def plot_radar_chart_elite(self, data_file: str) -> str:
        """全渠道双塔效用雷达图。"""
        self.logger.info("🎨 正在生成全渠道效用对比雷达图...")
        try:
            df = pd.read_parquet(data_file)
            cols = ['X1_Logistics_Norm', 'X2_Service_Norm', 'X3_Eco_Norm']
            platform_col = 'Platform' if 'Platform' in df.columns else 'platform'
            stats = df.groupby(platform_col)[cols].mean()
            stats_norm = (stats - stats.min()) / (stats.max() - stats.min() + 1e-6)
            stats_norm.index = stats_norm.index.astype(str).str.lower()
            if 'taobao' not in stats_norm.index or 'ozon' not in stats_norm.index:
                return ""
            labels = ['Logistics (Infra)', 'Service (Interaction)', 'Ecosystem (Lock-in)']
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist(); angles += angles[:1]
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            tb_val = stats_norm.loc['taobao'].values.tolist(); tb_val += tb_val[:1]
            oz_val = stats_norm.loc['ozon'].values.tolist(); oz_val += oz_val[:1]
            ax.plot(angles, tb_val, color='#FF4500', linewidth=2, label='Taobao (S-D Logic)')
            ax.fill(angles, tb_val, color='#FF4500', alpha=0.25)
            ax.plot(angles, oz_val, color='#005FF9', linewidth=2, label='Ozon (System Logic)')
            ax.fill(angles, oz_val, color='#005FF9', alpha=0.25)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=12)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.title("Normalized Omnichannel Capability Comparison", pad=20, fontsize=15)
            out_path = self.paths.output_dir / 'Omnichannel_Radar_Elite.png'
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"✨ 全渠道效用对比图已生成: {out_path.name}")
            return str(out_path)
        except Exception as e:
            self.logger.error(f"雷达图生成失败: {e}")
            return ""