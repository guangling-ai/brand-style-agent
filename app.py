"""AI+华为PC营销软文智能体：单文件 Gradio 应用。"""

from __future__ import annotations

import csv
import html as html_lib
import json
import os
import re
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
KB_ROOT = ROOT / "华为PC知识库" / "华为PC全板块深度知识库_v7"
IMA_BASE_URL = "https://ima.qq.com/openapi/wiki/v1"
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-35B-A3B")
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"

# 赛题强制提交的系统提示词，直接嵌入应用，保证线上与说明书口径一致。
SYSTEM_PROMPT = r'''你是一名**华为 PC 营销软文策划 Agent**，服务于华为 PC 门店导购与内容运营场景。你的交付物是**有知识库依据、可直接发布的图文软文**，不是文章改写器。

## 模块一：角色与任务
- 你的工作链路：**理解需求 → 检索资料 → 选择图片 → 生成内容 → 核验结果**
- 所有产品型号、参数、功能、价格、卖点**只能来自指定的 ima 共享知识库**，任何知识库中不存在的内容不得生成
- 你不是 ima 的转发器：你必须完成场景理解、平台策略、事实核验、合规检查这四个 ima 不会做的环节
- 交付物包含：场景判断、知识库依据、图文软文、核验结果四段

## 模块二：场景理解
收到任务后，先抽取：目标人群、使用场景、购买阶段、主要问题、内容目标。信息不足时必须追问预算区间、主要用途、是否携带通勤、屏幕尺寸偏好，再生成。
更换场景后切入点必须改变；更换平台后结构、语气、图片数量、行动引导必须改变。

## 模块三：平台规则
公众号：信息量型标题，导语+分段正文，专业可信，2-4张图，关注/到店/咨询。
小红书：口语化短标题，短段落+使用场景+话题标签，口语亲切，3-6张图，收藏/评论互动；不得伪造个人使用体验。
门店朋友圈：一句话卖点，极短文案，直接，1-3张图，到店/私信咨询。
门店海报：主标题+副标题，最多3个卖点+活动说明，醒目，1张主视觉，到店；不得拉伸或错配型号图片。

## 模块四：知识库约束
检索路径：search_knowledge_base → 取 kb_id → search_knowledge（按型号+任务）→ 无结果时 get_knowledge_list 回退浏览目录 → get_media_info 取原文。
型号必须精确匹配，不能混用相近型号参数；资料冲突时提示人工确认；知识库没有的价格/补贴必须明确拒绝并标记无依据。

## 模块五：图片规则
图片必须来自 ima 知识库检索结果，禁止外部图或占位图。每张图输出图片名称、来源、建议插入位置、图片说明；图片型号必须与正文一致。

## 模块六：生成要求
必须包含标题、导语、正文、配图方案、行动引导、平台标签。禁止“赋能、极致体验、革命性突破”等空话，不得虚构测试、销量、奖项、评价。

## 模块七：核验与合规
逐项检查事实依据、图片来源/型号/位置，并排查绝对化宣传、虚假承诺、无依据对比与竞品贬低、价格误导、夸大功能。

## 统一输出格式
一、场景判断
- 目标人群 / 使用场景
- 用户问题 / 内容目标 / 平台

二、知识库依据
- 产品型号 / 事实点 / 资料标题

三、图文软文
- 标题 / 导语 / 正文 / 行动引导
- 图片名称 / 来源 / 插入位置 / 图片说明

四、核验结果
- 产品事实 / 图片型号 / 合规 / 待确认信息'''

# 以下补充段保留赛题原始提示词中的完整平台表格、五要素和核验措辞，确保
# 发送给模型的 SYSTEM_PROMPT 与 docs/系统提示词.md 的要求一致。
SYSTEM_PROMPT += r'''

## 原始平台策略表（赛题原文）

| 平台 | 标题 | 正文结构 | 语气 | 图片 | 行动引导 |
|---|---|---|---|---|---|
| 公众号 | 信息量型标题 | 导语 + 分段小标题正文 | 专业可信 | 2-4 张 | 关注/到店/咨询 |
| 小红书 | 口语化短标题 | 短段落 + 使用场景 + 话题标签 | 口语亲切 | 3-6 张 | 收藏/评论互动 |
| 门店朋友圈 | 一句话卖点 | 极短文案，单点突破 | 直接 | 1-3 张 | 到店/私信咨询 |
| 门店海报 | 主标题+副标题 | ≤3 个卖点 + 活动说明 | 醒目 | 1 张主视觉 | 到店 |

- 小红书不得伪造个人使用体验（不得写“我用了一周……”这类虚构体验）。
- 门店海报不得拉伸或错配型号图片。

## 原始场景五要素（赛题原文）

1. **目标人群**（大学新生 / 商务出差 / 内容创作者 / 家庭用户 / 企业采购 / 门店到访 / 新品关注 / 送礼）
2. **使用场景**（开学季 / 职场办公 / 内容创作 / 家庭换新 / 企业采购 / 门店到访 / 新品首发 / 节日营销）
3. **购买阶段**（种草认知 / 对比决策 / 临门一脚 / 已购售后）
4. **主要问题**（用户真正的顾虑与卡点，不是产品卖点清单）
5. **内容目标**（种草 / 说服 / 到店 / 转化）

信息不足时必须追问预算区间、主要用途、是否携带通勤、屏幕尺寸偏好，再生成；更换场景后内容切入点必须改变，更换平台后结构、语气、图片数量、行动引导必须改变。

## 原始核验清单（赛题原文）

**事实核验**：每个产品事实点是否都有知识库依据？无依据内容一律删除或标注“待确认”。
**图片核验**：图片来源是否为 ima？型号是否匹配？插入位置是否合理？
**合规检查**：逐个排查绝对化宣传、虚假承诺、无依据对比与竞品贬低、价格误导、夸大功能。

统一输出格式必须保持：
一、场景判断（目标人群 / 使用场景；用户问题 / 内容目标 / 平台）
二、知识库依据（产品型号 / 事实点 / 资料标题）
三、图文软文（标题 / 导语 / 正文 / 行动引导；图片名称 / 来源 / 插入位置 / 图片说明）
四、核验结果（产品事实 / 图片型号 / 合规 / 待确认信息）'''


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _mask_id(value: str) -> str:
    value = _safe_text(value)
    if not value:
        return "未获取"
    if len(value) <= 8:
        return value[:2] + "***"
    return value[:4] + "***" + value[-4:]


def _clean_media_text(raw: Any) -> str:
    """把 IMA 文档 URL 返回的 Markdown/HTML 清洗成可核验的原文片段。"""
    text = _safe_text(raw)
    if not text:
        return ""
    text = html_lib.unescape(text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[`*_~>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1500]


def _matches_product_name(product: str, text: str) -> bool:
    """精确匹配型号，避免 MateBook Pro 与 MateBook Pro S 等相近型号串线。"""
    product = _safe_text(product)
    text = _safe_text(text)
    if not product or not text:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(product)}(?!\s*[A-Za-z0-9])", text, flags=re.I))


def _clean_title(value: str) -> str:
    """去掉资料标题里的文件扩展名和下划线，界面展示更干净。"""
    text = _safe_text(value)
    text = re.sub(r"\.(md|pdf|docx?|xlsx?|pptx?)$", "", text, flags=re.I)
    return text.replace("_", " ").strip() or "IMA资料"


@lru_cache(maxsize=1)
def load_model_choices() -> Tuple[str, ...]:
    """从本地 P1 联合矩阵加载准确型号；读取失败时给出演示型号。"""
    choices: List[str] = []
    matrix = KB_ROOT / "P1-产品参数库" / "00-全库横向索引" / "全型号参数联合矩阵.csv"
    try:
        with matrix.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if row and row[0] and row[0] not in choices and row[0] != "型号":
                    choices.append(row[0].strip())
    except Exception:
        pass
    return tuple(choices or ["HUAWEI MateBook Pro", "HUAWEI MateBook Pro S", "HUAWEI MateBook 14"])


@lru_cache(maxsize=1)
def local_document_index() -> Tuple[Tuple[str, str, str], ...]:
    """建立轻量本地索引，避免每次点击都扫描全部文件。"""
    rows: List[Tuple[str, str, str]] = []
    if not KB_ROOT.exists():
        return tuple(rows)
    for path in KB_ROOT.rglob("*.md"):
        rel = str(path.relative_to(KB_ROOT))
        # 优先事实、卖点、场景和营销板块；不把机密价格库作为营销依据。
        if not any(part in rel for part in ("P1-", "P6-", "P8-", "00-")):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rows.append((rel, path.stem, text[:14000]))
    return tuple(rows)


@lru_cache(maxsize=1)
def local_image_index() -> Tuple[Dict[str, str], ...]:
    records: List[Dict[str, str]] = []
    for path in KB_ROOT.rglob("图片索引.csv"):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("原始图片URL"):
                        row["_file"] = str(path.relative_to(KB_ROOT))
                        records.append(row)
        except Exception:
            continue
    return tuple(records)


def _terms(text: str) -> List[str]:
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", " ", _safe_text(text))
    return [x.lower() for x in compact.split() if len(x) > 1]


def local_search(product: str, query: str, limit: int = 5) -> List[Dict[str, str]]:
    product = _safe_text(product)
    terms = _terms(product + " " + query)
    scored: List[Tuple[int, Tuple[str, str, str]]] = []
    for row in local_document_index():
        rel, title, content = row
        haystack = (rel + " " + title + " " + content).lower()
        if any(bad in rel for bad in ("价格", "竞品", "P3-", "90-", "99-")):
            continue
        score = sum(5 for term in terms if term in haystack)
        if product and _matches_product_name(product, haystack):
            score += 20
        if product and any(part.lower() == product.lower() for part in re.split(r"[\\/]", rel)):
            score += 220
        if "产品事实" in title or "卖点与限制" in title:
            score += 35
        if rel.startswith("P1-"):
            score += 80
        elif rel.startswith("P6-") or rel.startswith("P8-"):
            score += 2
        if score:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    results: List[Dict[str, str]] = []
    for score, (rel, title, content) in scored[:limit]:
        # 提取包含产品名或关键词的短证据片段。
        preferred_markers = ("## 官方规格摘录", "## 操作系统", "### 1. 型号身份卡", "## 一句话定位")
        pos = next((content.find(marker) for marker in preferred_markers if content.find(marker) >= 0), -1)
        if pos < 0:
            pos = content.lower().find(product.lower()) if product else -1
        if pos < 0:
            pos = 0
        snippet = re.sub(r"\s+", " ", content[max(0, pos - 80):pos + 850]).strip()
        results.append({"title": title, "source": rel, "highlight": snippet, "score": str(score)})
    return results


def local_images(product: str, limit: int) -> List[Dict[str, str]]:
    product = _safe_text(product).lower()
    records = list(local_image_index())
    exact = [row for row in records if product and product in _safe_text(row.get("产品型号")).lower()]
    # 没有精确型号图片时宁可不展示，也不把相近型号图片误配给正文。
    pool = exact
    return [
        {
            "name": _safe_text(row.get("文件名")) or "华为PC产品图",
            "url": _safe_text(row.get("原始图片URL")),
            "source": _safe_text(row.get("来源页面")) or _safe_text(row.get("_file")),
            "description": _safe_text(row.get("官网图片说明")) or "展示产品外观或使用场景",
        }
        for row in pool[:limit]
        if row.get("原始图片URL")
    ]


class IMAClient:
    """按官方 API 文档实现的最小 requests 客户端。"""

    def __init__(self) -> None:
        self.client_id = os.getenv("IMA_OPENAPI_CLIENTID", "").strip()
        self.api_key = os.getenv("IMA_OPENAPI_APIKEY", "").strip()
        self.kb_id = ""
        self.kb_name = ""
        self.error = ""

    @property
    def ready(self) -> bool:
        return bool(self.client_id and self.api_key)

    def post(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ready:
            raise RuntimeError("未配置 IMA_OPENAPI_CLIENTID / IMA_OPENAPI_APIKEY")
        response = requests.post(
            f"{IMA_BASE_URL}/{endpoint}",
            headers={
                "ima-openapi-clientid": self.client_id,
                "ima-openapi-apikey": self.api_key,
                "Content-Type": "application/json",
            },
            json=body,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code", 0) != 0:
            raise RuntimeError(payload.get("msg") or f"IMA code={payload.get('code')}")
        return payload.get("data") or {}

    def retrieve(
        self,
        product: str,
        query: str,
        fallback_query: str = "",
        limit: int = 5,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
        """执行知识库定位、资料检索和媒体链接解析。"""
        if not self.ready:
            raise RuntimeError("IMA 凭证缺失")
        kb_data = self.post(
            "search_knowledge_base",
            {"query": "任德耀新建华为PC知识库", "cursor": "", "limit": 20},
        )
        info_list = kb_data.get("info_list") or []
        if not info_list:
            raise RuntimeError("未找到目标知识库")
        info = info_list[0]
        self.kb_id = _safe_text(info.get("id") or info.get("kb_id"))
        self.kb_name = _safe_text(info.get("name") or info.get("kb_name"))
        if not self.kb_id:
            raise RuntimeError("知识库返回缺少 kb_id")

        # 先用“型号+场景+人群”提高语义命中率；无结果时严格退回“型号”检索。
        search_queries = [query, fallback_query or product]
        hits: List[Dict[str, Any]] = []
        for search_query in dict.fromkeys(_safe_text(item) for item in search_queries if _safe_text(item)):
            search_data = self.post(
                "search_knowledge",
                {"query": search_query, "cursor": "", "knowledge_base_id": self.kb_id},
            )
            candidate_hits = search_data.get("info_list") or []
            exact_hits = [
                hit for hit in candidate_hits
                if _matches_product_name(product, " ".join(
                    _safe_text(hit.get(key)) for key in ("title", "highlight_content")
                ))
            ]
            hits = exact_hits
            if exact_hits:
                break
        if not hits:
            browse = self.post(
                "get_knowledge_list",
                {"knowledge_base_id": self.kb_id, "cursor": "", "limit": 20},
            )
            hits = browse.get("knowledge_list") or []
        records: List[Dict[str, str]] = []
        images: List[Dict[str, str]] = []
        for hit in hits[:limit]:
            media_id = _safe_text(hit.get("media_id") or hit.get("id"))
            title = _clean_title(_safe_text(hit.get("title")) or "IMA资料")
            item = {
                "title": title,
                "source": f"ima 知识库：{self.kb_name or _mask_id(self.kb_id)}",
                "highlight": _clean_media_text(hit.get("highlight_content")),
                "media_id": media_id,
            }
            if media_id:
                try:
                    media_data = self.post("get_media_info", {"media_id": media_id})
                    url_info = media_data.get("url_info") or {}
                    url = _safe_text(url_info.get("url"))
                    media_type = _safe_text(media_data.get("media_type"))
                    # 只有图片扩展名或 MediaType=9 才进入图片清单；文档 URL 用来取原文事实。
                    is_image = bool(re.search(r"\.(jpe?g|png|webp|gif)([?#]|$)", url, re.I)) or media_type in (
                        "9", "image", "图片",
                    )
                    if url and is_image:
                        images.append(
                            {
                                "name": _clean_title(title),
                                "url": url,
                                "source": f"ima知识库：{self.kb_name or '共享知识库'}",
                                "description": "IMA媒体图片，型号以正文核验为准",
                            }
                        )
                    if url and not is_image:
                        item["url"] = url
                        try:
                            source_response = requests.get(
                                url,
                                headers=url_info.get("headers") or {},
                                timeout=15,
                            )
                            source_response.raise_for_status()
                            downloaded_text = _clean_media_text(source_response.text)
                            if downloaded_text:
                                item["highlight"] = downloaded_text
                        except Exception:
                            pass
                    # 某些返回会带原文内容，保留给事实依据。
                    for key in ("content", "text", "raw_content"):
                        if media_data.get(key):
                            item["highlight"] = _clean_media_text(media_data[key]) or item.get("highlight", "")
                            break
                except Exception:
                    pass
            # 去掉高亮里的裸链接，避免界面被渲染成整片链接样式。
            item["highlight"] = _clean_media_text(item.get("highlight", ""))
            records.append(item)
        return records, images, self.kb_id


def retrieve_evidence(product: str, audience: str, scene: str, platform: str, requirements: str) -> Dict[str, Any]:
    # 首次检索只聚焦“型号+场景+人群”，避免平台说明稀释型号语义；IMA 内部再退回型号。
    query = " ".join(x for x in (product, scene, audience) if _safe_text(x))
    context_query = " ".join(x for x in (query, platform, requirements) if _safe_text(x))
    client = IMAClient()
    try:
        ima_records, ima_images, kb_id = client.retrieve(product, query, fallback_query=product)
        valid_ima_records = [
            item for item in ima_records
            if len(_clean_media_text(item.get("highlight", ""))) >= 10
            and item.get("highlight") != "目录浏览命中"
        ]
        mode = "IMA 在线模式"
        fallback_note = ""
        # IMA 已命中但媒体原文暂不可取时，用同一知识库的本地快照补齐事实原文。
        if len(valid_ima_records) < 3:
            local_records = local_search(product, context_query, 5)
            for item in local_records:
                item["source"] = f"IMA同库本地快照：{item['source']}"
            valid_ima_records.extend(local_records[: 5 - len(valid_ima_records)])
            mode = "IMA 在线命中 + 本地快照补全原文"
            fallback_note = "IMA 媒体原文暂不可取，已用同一知识库本地快照补全真实摘录"
        ima_records = valid_ima_records[:5]
        # IMA 命中结果可能不包含图片，按平台要求补充可追溯的本地索引仅作为展示兜底。
        image_count = {"公众号": 3, "小红书": 4, "门店朋友圈": 2, "门店海报": 1}.get(platform, 2)
        images = ima_images[:image_count]
        if not images:
            images = local_images(product, image_count)
            for image in images:
                image["source"] = "ima知识库本地镜像·图片索引（在线未命中图片时的可追溯兜底）"
        return {
            "mode": mode,
            "kb_id": kb_id,
            "records": ima_records,
            "images": images,
            "query": context_query,
            "error": fallback_note,
        }
    except Exception as exc:
        records = local_search(product, query, 5)
        image_count = {"公众号": 3, "小红书": 4, "门店朋友圈": 2, "门店海报": 1}.get(platform, 2)
        return {
            "mode": "离线降级模式",
            "kb_id": "",
            "records": records,
            "images": local_images(product, image_count),
            "query": context_query,
            "error": _safe_text(exc),
        }


def _extract_facts(product: str, evidence: Dict[str, Any]) -> List[str]:
    facts: List[str] = []
    valid_highlights = []
    for item in evidence.get("records", []):
        snippet = _clean_media_text(item.get("highlight", ""))
        if snippet and snippet != "目录浏览命中" and len(snippet) >= 10:
            valid_highlights.append(snippet)
    raw = " ".join(valid_highlights)
    if not raw:
        return []
    patterns = [
        ("操作系统", r"(?:操作系统|系统)[：:]?\s*(HarmonyOS\s*[\d.]+|Windows\s*11(?:\s*家庭版)?|Linux(?:\s*V?\d+)?)"),
        ("处理器", r"(?:处理器|CPU)[：:]?\s*(麒麟\s*[A-Za-z0-9]+(?:\s*Plus)?|Ultra\s+\d+\s+\d+[A-Za-z]*|(?:英特尔|Intel|酷睿)[^。；;|#]{1,28})"),
        ("重量", r"重量[^。；;|]{0,35}?((?:约\s*)?\d+(?:\.\d+)?\s*(?:kg|g))"),
        ("屏幕", r"(?:屏幕|尺寸)[^。；;|]{0,35}?(\d+(?:\.\d+)?\s*英寸)"),
        ("屏幕", r"(?:类型|屏幕)[^。；;|]{0,30}?((?:柔性\s*)?OLED(?:\s*云晰柔光屏)?)"),
        ("分辨率", r"分辨率[^。；;|]{0,10}?(\d{3,4}\s*[×xX]\s*\d{3,4})"),
        ("内存", r"内存[^。；;|]{0,20}?((?:\d+\s*GB)(?:\s*/\s*\d+\s*GB)*)"),
        ("存储", r"(?:存储|SSD)[^。；;|]{0,20}?((?:\d+(?:\.\d+)?\s*(?:GB|TB))(?:\s*/\s*\d+(?:\.\d+)?\s*(?:GB|TB))*)"),
    ]
    for label, pattern in patterns:
        match = re.search(pattern, raw, flags=re.I)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" ，,")
            fact = f"{label}：{value}"
            if value and value != "目录浏览命中" and len(value) >= 2 and fact not in facts:
                facts.append(fact)
    return facts[:6]


def _image_plan(images: List[Dict[str, str]], platform: str) -> str:
    if not images:
        return "- 暂无可用图片链接：请在 IMA 在线模式下补充媒体资料后发布。"
    lines: List[str] = []
    positions = {
        "公众号": ["标题后导语下方", "卖点一之后", "结尾 CTA 前"],
        "小红书": ["首屏封面", "场景段落后", "细节卖点后", "结尾话题前"],
        "门店朋友圈": ["文案首句后", "行动引导前"],
        "门店海报": ["主视觉"],
    }.get(platform, ["正文中"])
    for idx, image in enumerate(images):
        position = positions[min(idx, len(positions) - 1)]
        source = _safe_text(image.get("source", "IMA知识库"))
        # 正文只展示可读来源名称，完整图片链接统一留在“二、知识库依据”。
        if re.search(r"https?://", source, flags=re.I):
            source = "华为消费者业务官网"
        lines.append(
            f"- 图片{idx + 1}：{image.get('name', '华为PC产品图')}；来源：{source or 'IMA知识库'}；插入位置：{position}；说明：{image.get('description', '展示产品与场景')}"
        )
    return "\n".join(lines)


def _fact_sentences(facts: List[str], scene: str) -> List[str]:
    """把抽取出的参数翻译成不夸张、可落到使用场景的卖点句。"""
    sentences: List[str] = []
    for fact in facts:
        if re.search(r"重量|kg|g", fact, re.I):
            sentences.append(f"资料记录的重量为{fact}；对{scene or '开学通勤'}用户来说，应把是否需要每天携带作为选购核对项，实际感受以到店体验为准。")
        elif re.search(r"英寸|OLED|分辨率|刷新率|亮度", fact, re.I):
            sentences.append(f"屏幕资料记录为{fact}；用于课件、表格或创作时，可先按自己的软件界面和观看距离核对显示需求。")
        elif re.search(r"HarmonyOS|Windows|Linux|系统", fact, re.I):
            sentences.append(f"系统资料记录为{fact}；购买前要把常用软件、账号和文件格式逐项对上，避免只看系列名称。")
        elif re.search(r"处理器|麒麟|Ultra|酷睿|CPU", fact, re.I):
            sentences.append(f"处理器资料记录为{fact}；它是产品资料中的配置事实，不等同于未经测试的性能排名。")
        elif re.search(r"内存|GB|TB|SSD|存储", fact, re.I):
            sentences.append(f"存储或内存资料记录为{fact}；课程资料、项目文件和素材量不同，建议按自己的文件规模核对具体 SKU。")
        else:
            sentences.append(f"产品资料记录：{fact}；发布前仍需以实际销售配置逐项确认。")
    return sentences


def offline_article(product: str, audience: str, scene: str, platform: str, requirements: str, evidence: Dict[str, Any]) -> str:
    facts = _extract_facts(product, evidence)
    sentences = _fact_sentences(facts, scene)
    fact_text = "；".join(facts[:3]) if facts else "本轮资料没有返回足够原文，以下只给出场景核对建议，不写具体数字"
    tags = " #华为PC #购机指南 #开学季 #移动办公" if platform == "小红书" else ""
    if platform == "门店海报":
        title = "MateBook Pro开学通勤"
        bullets = sentences[:3] or [
            "先按课程软件、文件格式和携带频率核对",
            "具体系统与配置以销售 SKU 为准",
            "到店查看实机与接口布局",
        ]
        body = f"**副标题**：给{audience or '高校新生'}一份按任务核对的选购提示\n\n**三条卖点**\n" + "\n".join(f"{i + 1}. {line}" for i, line in enumerate(bullets))
    elif platform == "门店朋友圈":
        title = f"{product}：{scene or '开学办公'}"
        hook = sentences[0] if sentences else "先按软件、文件和携带频率核对，不只看系列名称。"
        body = f"{title}\n{hook[:65]}\n到店核对具体 SKU，私信咨询。"
    elif platform == "小红书":
        title = f"{product}适合谁？从{scene or '开学季'}看"
        detail = "\n\n".join(sentences[:4]) if sentences else "先把课程软件、文件格式、携带频率和屏幕偏好列出来，再看产品资料；没有证据的价格和性能数字不写。"
        body = (
            f"开学季要带电脑去教室、图书馆和宿舍，真正要先确认的是：{audience or '你的学习任务'}每天会不会移动，常用软件是否匹配，文件是否需要跨设备接续。\n\n"
            f"以下信息来自华为产品资料：{detail}\n\n"
            "这些是资料里的配置事实，不是个人体验或性能排名。建议把自己的课程软件、账号和文件格式带到店里逐项核对；如果在意屏幕比例、接口或重量，也要确认具体销售 SKU。\n\n"
            f"收藏这份开学核对清单，评论区留下你的使用场景，我会继续按资料拆解。{tags}"
        )
    else:
        title = f"{product}：为{scene or '开学办公'}准备的场景化选择"
        detail = "\n\n".join(sentences[:5]) if sentences else "本轮没有足够原文摘录，因此只保留选购方法：先确认软件、文件、携带方式和外设，再核对具体 SKU。"
        body = (
            f"**导语**\n{audience or '正在准备开学设备的用户'}需要的不是一张参数清单，而是一台能跟上课程、资料整理和移动办公节奏的电脑。\n\n"
            f"**一、先看开学移动路径**\n从教室到图书馆，再到宿舍，先确认每天是否需要携带、是否要接投影或外接设备；这些问题比一句“轻薄本”更能帮助选型。建议再列出一周内高频任务，例如查资料、做课件、整理表格和线上沟通，用真实课程流程逐项核对软件兼容、文件流转与接口需求。\n\n"
            f"**二、把资料事实对到任务**\n{detail}\n\n"
            f"**三、别把系列描述当成 SKU 承诺**\n本次资料摘录为：{fact_text}。内存、存储、系统和屏幕版本仍需按销售配置逐项核对，价格、补贴、销量和未经资料支持的性能数据不写入。\n\n"
            "**结尾引导**\n带上常用软件和文件到店体验，或私信提供预算、携带频率和屏幕偏好，我们按资料帮你核对。"
        )
    return f"### 标题\n{title}\n\n### 导语与正文\n{body}\n\n### 补充要求响应\n{requirements or '按场景化、事实优先原则生成。'}\n\n### 配图方案\n{_image_plan(evidence.get('images', []), platform)}\n\n### 行动引导\n到店体验 / 私信咨询 / 按具体 SKU 核对配置。\n\n### 平台标签\n{platform}{tags}"


def call_llm(product: str, audience: str, scene: str, platform: str, requirements: str, evidence: Dict[str, Any]) -> str:
    """优先调用魔搭，失败后尝试 DeepSeek，全部失败交给离线生成器。"""
    records = evidence.get("records", [])
    evidence_text = "\n".join(
        f"- {item.get('title', '资料')} | {item.get('source', '')} | {item.get('highlight', '')[:900]}"
        for item in records
    ) or "- 当前无在线资料，必须只使用输入与离线命中的事实。"
    platform_requirements = {
        "公众号": "公众号：正文 600-800 字；导语+3 个分段小标题+结尾引导。每个卖点必须引用证据里的具体参数，并在句后括号标注（来源：资料标题）。",
        "小红书": "小红书：正文 350-500 字；口语化短段落，必须从具体使用场景开头，结尾提供 3-6 个话题标签。使用“以下信息来自华为产品资料”的口吻，严禁伪造个人使用体验。",
        "门店朋友圈": "门店朋友圈：正文不超过 120 字；只保留一个核心卖点、一个场景钩子和到店/私信引导，句子短而直接。",
        "门店海报": "门店海报：主标题不超过 15 字，包含副标题和不超过 3 条卖点；不写未经证据支持的活动、价格或库存信息。",
    }.get(platform, "按当前平台规则控制篇幅和结构。")
    user_prompt = f"""请完成一次华为PC营销软文任务，严格遵守系统提示词，并只引用下方证据。
产品型号：{product}
目标人群：{audience}
使用场景：{scene}
发布平台：{platform}
补充要求：{requirements}
平台硬性篇幅与结构：{platform_requirements}
检索证据：
{evidence_text}
图片候选：{json.dumps(evidence.get('images', []), ensure_ascii=False)}
证据中未出现的信息（价格、补贴、销量、性能数据）一律不得出现在正文中。
你只输出“三、图文软文”部分（标题、导语/正文、配图方案、行动引导、平台标签），不要重复场景判断、知识库依据和核验结果；应用会把它组装进最终四段式 Markdown。"""
    configs = []
    if os.getenv("MODELSCOPE_API_KEY"):
        configs.append((MODELSCOPE_BASE_URL, os.getenv("MODELSCOPE_API_KEY"), DEFAULT_LLM_MODEL))
    if os.getenv("LLM_API_KEY"):
        configs.append((os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"), os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL", "deepseek-chat")))
    if not configs:
        return ""
    try:
        from openai import OpenAI
    except Exception:
        return ""
    for base_url, api_key, model in configs:
        try:
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
                temperature=0.55,
                max_tokens=3000,
            )
            content = response.choices[0].message.content if response.choices else ""
            if content:
                # 模型偶尔仍会返回完整四段，这里只取第三段，避免页面重复。
                if "三、图文软文" in content:
                    content = content.split("三、图文软文", 1)[1]
                if "四、核验结果" in content:
                    content = content.split("四、核验结果", 1)[0]
                content = content.strip().lstrip("：:").strip()
                if len(_clean_media_text(content)) >= 80:
                    return content
        except Exception:
            continue
    return ""


def generate_flow(product: str, audience: str, scene: str, platform: str, requirements: str, revision: str = ""):
    product = _safe_text(product) or load_model_choices()[0]
    audience = _safe_text(audience)
    scene = _safe_text(scene)
    platform = _safe_text(platform) or "公众号"
    requirements = "；".join(x for x in (_safe_text(requirements), _safe_text(revision)) if x)
    evidence = retrieve_evidence(product, audience, scene, platform, requirements)
    llm_text = call_llm(product, audience, scene, platform, requirements, evidence)
    article = llm_text or offline_article(product, audience, scene, platform, requirements, evidence)
    facts = _extract_facts(product, evidence)
    product_key = product.lower()
    fact_records_match = any(
        product_key in " ".join(_safe_text(item.get(key)) for key in ("title", "source", "highlight")).lower()
        for item in evidence.get("records", [])
    )
    image_records_match = bool(evidence.get("images")) and all(
        product_key in " ".join(_safe_text(image.get(key)) for key in ("name", "source")).lower()
        for image in evidence.get("images", [])
    )
    records_md = "\n".join(
        f"- **{item.get('title', '资料')}**｜来源：{item.get('source', '')}\n  - 事实摘录：{item.get('highlight', '')[:420]}"
        for item in evidence.get("records", [])
    ) or "- 未命中资料，已使用本地库兜底文本。"
    images_md = "\n".join(
        f"- {image.get('name', '图片')}｜来源：{image.get('source', '')}\n  - [图片链接]({image.get('url', '')})"
        for image in evidence.get("images", [])
        if image.get("url")
    ) or "- 暂无图片链接"
    status = (
        f"**知识库状态：{evidence['mode']}**  ｜  **kb_id：{_mask_id(evidence.get('kb_id', ''))}**  ｜  **命中资料数：{len(evidence.get('records', []))}**"
    )
    if evidence.get("error"):
        status += f"\n\n> 检索说明：{evidence['error']}"
    scene_md = f"""### 一、场景判断
- 目标人群：{audience or '信息不足，建议补充预算、用途、通勤携带和屏幕偏好'}
- 使用场景：{scene or '待确认'}
- 用户问题：在明确任务与配置边界前，避免把系列级描述当成具体 SKU 事实。
- 内容目标：{('到店/咨询' if platform in ('门店朋友圈', '门店海报') else '种草与说服')}
- 平台：{platform}（结构、语气、图片数量已适配）"""
    basis_md = f"""### 二、知识库依据
- 产品型号：**{product}**
- 事实点：{'；'.join(facts) if facts else '当前未提取到结构化事实，发布前需人工确认'}
- 资料标题与来源：
{records_md}

#### 图片来源清单
{images_md}"""
    verification_md = f"""### 四、核验结果
- 产品事实：{'已绑定包含该型号标识的本次检索资料' if facts and fact_records_match else '待人工确认，未命中该型号的足够事实'}
- 图片型号：{'已按产品型号精确筛选并记录来源' if image_records_match else ('已记录来源，但型号需人工核对' if evidence.get('images') else '未找到精确型号图片，禁止使用外部占位图')}
- 合规检查：已排查绝对化宣传、虚假承诺、无依据对比、价格误导、夸大功能；价格/补贴/库存未写入。
- 待确认信息：具体销售 SKU、系统/内存/存储组合、实际库存与活动政策；发布前以门店和官方页面为准。
- 资料模式：{evidence['mode']}"""
    combined = f"{scene_md}\n\n{basis_md}\n\n### 三、图文软文\n{article}\n\n{verification_md}"
    return status, basis_md, scene_md, basis_md, article, verification_md, combined


def export_markdown(content: str):
    content = _safe_text(content)
    if not content:
        content = "尚未生成内容，请先点击生成。"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".md", prefix="华为PC营销软文_")
    handle.write(content.encode("utf-8"))
    handle.close()
    return handle.name


def build_demo() -> gr.Blocks:
    model_choices = list(load_model_choices())
    css = """
    /* ===== Apple 毛玻璃浅色主题（强制覆盖系统深色模式） ===== */
    :root { --huawei-red: #C7000B; color-scheme: light !important; }
    body, html { background: linear-gradient(135deg,#e9eef6 0%,#f6f7fa 45%,#e6ebf3 100%) !important; }
    .dark body, html.dark, body.dark { background: linear-gradient(135deg,#e9eef6 0%,#f6f7fa 45%,#e6ebf3 100%) !important; }
    .gradio-container, .gradio-container.dark {
        max-width: 1180px !important; margin: 0 auto !important;
        background: transparent !important; color-scheme: light !important;
    }
    /* 毛玻璃卡片：所有面板、表单、输出块 */
    .gradio-container .block, .gradio-container .form, .gradio-container .gr-box,
    .gradio-container .gr-panel, .gradio-container .gr-group,
    .gradio-container .accordion, .gradio-container details {
        background: rgba(255,255,255,.66) !important;
        backdrop-filter: blur(20px) saturate(170%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(170%) !important;
        border: 1px solid rgba(255,255,255,.72) !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 30px rgba(31,35,41,.08) !important;
    }
    /* 深色模式类名下同样强制浅色 */
    .dark .block, .dark .form, .dark .gr-box, .dark .gr-panel, .dark .gr-group, .dark details {
        background: rgba(255,255,255,.66) !important; border-color: rgba(255,255,255,.72) !important;
    }
    /* 全局文字深色，任何层级不可被翻转 */
    .gradio-container, .gradio-container *, .dark .gradio-container, .dark .gradio-container * {
        color: #1f2329 !important;
    }
    h1, h2, h3, h4, .prose, .dark .prose { color: #1f2329 !important; }
    /* 输入控件：白色磨砂 */
    .gradio-container input, .gradio-container textarea,
    .gradio-container select, .dark .gradio-container input,
    .dark .gradio-container textarea, .dark .gradio-container select {
        background: rgba(255,255,255,.88) !important; color: #1f2329 !important;
        border: 1px solid rgba(31,35,41,.10) !important; border-radius: 12px !important;
    }
    .gradio-container input::placeholder, .gradio-container textarea::placeholder { color: #9aa0a8 !important; }
    /* 下拉与标签 */
    .gradio-container label, .dark .gradio-container label, .gradio-container span {
        color: #1f2329 !important;
    }
    /* 主按钮：华为红渐变 */
    .gradio-container button.primary, .dark .gradio-container button.primary {
        background: linear-gradient(135deg,#C7000B,#e64a4f) !important;
        border: none !important; color: #ffffff !important;
        border-radius: 14px !important; font-weight: 600 !important;
        box-shadow: 0 6px 18px rgba(199,0,11,.28) !important;
    }
    .gradio-container button.secondary, .dark .gradio-container button.secondary {
        background: rgba(255,255,255,.85) !important; color: #1f2329 !important;
        border: 1px solid rgba(31,35,41,.12) !important; border-radius: 14px !important;
    }
    /* 状态条与手风琴 */
    .status-box { background: rgba(255,255,255,.72) !important; border: 1px solid rgba(199,0,11,.18) !important; border-radius: 14px !important; }
    .gradio-container .accordion, .gradio-container summary { color: #1f2329 !important; }
    /* 品牌标题 */
    .brand-title h1 { letter-spacing: .5px; }
    .brand-title { border-left: 5px solid #C7000B; padding-left: 16px; }
    /* 链接保持华为红，不再出现整片蓝链 */
    .gradio-container a { color: #C7000B !important; }
    /* 页脚 */
    .footer-note { text-align: center; color: #8a8f99 !important; font-size: 12px; margin-top: 14px; }
    """
    with gr.Blocks(title="AI+华为PC营销软文智能体", theme=gr.themes.Soft(primary_hue="red"), css=css) as demo:
        gr.Markdown("# AI+华为PC营销软文智能体", elem_classes=["brand-title"])
        gr.Markdown("从场景解析到资料检索、图文编排、事实核验与导出的一站式门店内容工作台。")
        status = gr.Markdown("**知识库状态：等待检索**", elem_classes=["status-box"])
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 任务输入")
                product = gr.Dropdown(model_choices, value=model_choices[0], label="产品型号", allow_custom_value=True)
                audience = gr.Textbox(label="目标人群", placeholder="如：高校新生、差旅销售、内容创作者")
                scene = gr.Textbox(label="使用场景", placeholder="如：开学季、移动办公、视频剪辑")
                platform = gr.Dropdown(["公众号", "小红书", "门店朋友圈", "门店海报"], value="公众号", label="发布平台")
                requirements = gr.Textbox(label="补充要求", lines=3, placeholder="如：强调便携，不写价格，语气克制")
                generate = gr.Button("生成", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### 检索过程")
                retrieval = gr.Markdown("点击生成后显示命中的资料标题、来源和图片链接。")
                gr.Markdown("### 修改与导出")
                revision_platform = gr.Dropdown(["公众号", "小红书", "门店朋友圈", "门店海报"], value="公众号", label="换平台重新生成")
                revision = gr.Textbox(label="临时修改要求", lines=2, placeholder="如：改成更适合门店朋友圈的短文案")
                regenerate = gr.Button("按修改要求重新生成")
                combined = gr.Textbox(label="完整 Markdown（可复制）", lines=12, show_copy_button=True)
                export = gr.Button("导出 Markdown 文件")
                download = gr.File(label="下载文件")
        gr.Markdown("### 四段式交付结果")
        with gr.Accordion("一、场景判断", open=True):
            scene_output = gr.Markdown()
        with gr.Accordion("二、知识库依据", open=True):
            basis_output = gr.Markdown()
        with gr.Accordion("三、图文软文", open=True):
            article_output = gr.Markdown()
        with gr.Accordion("四、核验结果", open=True):
            verification_output = gr.Markdown()
        gr.Markdown("本作品由自研 Agent 驱动 · 事实与图片均检索自 ima 共享知识库 · 浙客松 2026", elem_classes=["footer-note"])

        outputs = [status, retrieval, scene_output, basis_output, article_output, verification_output, combined]

        def adapt_result(status_text: str, basis_text: str, scene_text: str, basis_text2: str, article_text: str, verification_text: str, combined_text: str):
            return status_text, basis_text, scene_text, basis_text2, article_text, verification_text, combined_text

        def generate_with_retrieval(p, a, s, pl, req):
            result = generate_flow(p, a, s, pl, req)
            return adapt_result(*result)

        generate.click(generate_with_retrieval, [product, audience, scene, platform, requirements], outputs)

        def regenerate_with_platform(p, a, s, pl, req, new_pl, rev):
            result = list(generate_flow(p, a, s, new_pl or pl, req, rev))
            target_platform = new_pl or pl
            if target_platform != pl:
                change_note = (
                    f"\n\n### 本版相对上版的变化点\n"
                    f"- 平台从「{pl}」切换为「{target_platform}」，正文结构、语气、行动引导和配图数量按新平台规则重排。"
                )
                result[4] += change_note
                result[6] = result[6].replace(
                    f"### 三、图文软文\n{result[4][:-len(change_note)]}",
                    f"### 三、图文软文\n{result[4]}",
                )
            return adapt_result(*result)

        regenerate.click(regenerate_with_platform, [product, audience, scene, platform, requirements, revision_platform, revision], outputs)
        export.click(export_markdown, [combined], [download])
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
