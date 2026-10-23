"""AI+华为PC营销软文智能体：单文件 Gradio 应用。"""

from __future__ import annotations

import base64
import csv
import html as html_lib
import json
import mimetypes
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlsplit

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
KB_ROOT = ROOT / "华为PC知识库" / "华为PC全板块深度知识库_v7"
IMA_BASE_URL = "https://ima.qq.com/openapi/wiki/v1"
DEFAULT_LLM_MODEL = os.getenv("MODELSCOPE_MODEL", "Qwen/Qwen3.5-35B-A3B")
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


def _is_competitor(item: Dict[str, Any]) -> bool:
    """标题/正文头部识别对照卡，统一供在线、本地和生成入口使用。"""
    title = _safe_text(item.get("title"))
    head = _safe_text(item.get("highlight")).split("\n", 1)[0][:160]
    path = _safe_text(item.get("source"))
    return bool(re.search(
        r"(?<![A-Za-z])(?:HP|Lenovo|Dell|Apple|MacBook|Xiaomi|REDMI|ThinkPad|ThinkBook|ASUS)(?![A-Za-z])|惠普|联想|戴尔|苹果|小米|华硕|竞品|对比|对照|×",
        title + " " + head, re.I,
    ) or re.search(r"P3-|竞品", path, re.I))


def _is_talk(item: Dict[str, Any]) -> bool:
    return bool(re.search(r"P2-|话术|沟通卡|参数转收益", " ".join(
        _safe_text(item.get(key)) for key in ("title", "source", "category")
    )))


def _is_own_fact(product: str, item: Dict[str, Any]) -> bool:
    if _is_competitor(item) or _is_talk(item):
        return False
    identity = " ".join(_safe_text(item.get(key)) for key in ("title", "source"))
    head = _safe_text(item.get("highlight"))[:200]
    return _matches_product_name(product, identity + " " + head) and bool(
        re.search(r"P1-|产品事实|参数|规格|发现卡|价格卡|卖点与限制", identity + " " + head)
    )


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
        if any(part.startswith(("P3-", "P4-", "90-", "99-")) for part in path.relative_to(KB_ROOT).parts):
            continue
        if not rel.startswith(("P1-", "P2-", "P6-", "P8-", "00-")):
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


def local_search(product: str, query: str, limit: int = 5, talks_only: bool = False) -> List[Dict[str, str]]:
    product = _safe_text(product)
    terms = _terms(product + " " + query)
    scored: List[Tuple[int, Tuple[str, str, str]]] = []
    for row in local_document_index():
        rel, title, content = row
        haystack = (rel + " " + title + " " + content).lower()
        if talks_only and not rel.startswith("P2-"):
            continue
        if _is_competitor({"title": title, "source": rel, "highlight": content}):
            continue
        if product and not _matches_product_name(product, rel + " " + content[:200]):
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
        snippet = content if rel.startswith(("P1-", "P2-")) else content[max(0, pos - 80):pos + 1500]
        urls = re.findall(r"https://(?:consumer|consumer\.huawei|www\.vmall)[^\s)]+", content)
        results.append({"title": title, "source": rel, "highlight": snippet, "score": str(score),
                        "url": urls[0] if urls else ""})
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
            timeout=12,
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
        include_talks: bool = False,
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
            hits.extend(hit for hit in exact_hits if hit not in hits)
            if any(not _is_competitor({"title": hit.get("title"), "highlight": hit.get("highlight_content")}) for hit in exact_hits):
                break
        if not hits:
            browse = self.post(
                "get_knowledge_list",
                {"knowledge_base_id": self.kb_id, "cursor": "", "limit": 20},
            )
            hits = browse.get("knowledge_list") or []
        # 资料类型单独检索，避免竞品卡占满前几个命中；话术检索失败不拖垮事实路径。
        for suffix in (["产品事实"] + (["卖点 话术"] if include_talks else [])):
            try:
                extra = self.post("search_knowledge", {
                    "query": f"{product} {suffix}", "cursor": "", "knowledge_base_id": self.kb_id,
                }).get("info_list") or []
                for hit in extra:
                    if hit not in hits and _matches_product_name(product, str(hit.get("title", "")) + " " + str(hit.get("highlight_content", ""))):
                        hit = dict(hit)
                        if suffix == "卖点 话术" and _is_talk({"title": hit.get("title", "")}):
                            hit["category"] = "P2-卖点话术库"
                        hits.append(hit)
            except Exception:
                continue
        records: List[Dict[str, str]] = []
        images: List[Dict[str, str]] = []
        for hit in hits[:limit + 6]:
            media_id = _safe_text(hit.get("media_id") or hit.get("id"))
            title = _clean_title(_safe_text(hit.get("title")) or "IMA资料")
            item = {
                "title": title,
                "source": f"ima 知识库：{self.kb_name or _mask_id(self.kb_id)}",
                "highlight": _clean_media_text(hit.get("highlight_content")),
                "media_id": media_id,
                "category": _safe_text(hit.get("category")),
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
                                timeout=8,
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
    needs_talks = platform in ("门店朋友圈", "门店海报") or bool(re.search(r"导购|到店|门店", scene))
    try:
        ima_records, ima_images, kb_id = client.retrieve(product, query, fallback_query=product, include_talks=needs_talks)
        valid_ima_records = [
            item for item in ima_records
            if len(_clean_media_text(item.get("highlight", ""))) >= 10
            and item.get("highlight") != "目录浏览命中"
        ]
        mode = "IMA 在线模式"
        fallback_note = ""
        # IMA 已命中但媒体原文暂不可取时，用同一知识库的本地快照补齐事实原文。
        if not any(_is_own_fact(product, item) and "产品事实" in item.get("title", "")
                   for item in valid_ima_records) or len(valid_ima_records) < 3:
            local_records = [item for item in local_search(product, context_query, 8) if _is_own_fact(product, item)][:3]
            for item in local_records:
                item["source"] = f"IMA同库本地快照：{item['source']}"
            valid_ima_records.extend(local_records)
            mode = "IMA 在线命中 + 本地快照补全原文"
            fallback_note = "IMA 未返回足够的本方产品事实卡原文，已用同一知识库 P1 本地快照补全并交叉校验"
        ima_records = valid_ima_records
        if needs_talks and not any(_is_talk(item) and not _is_competitor(item) for item in ima_records):
            for item in local_search(product, context_query + " 门店顾客 卖点 话术", 1, talks_only=True):
                item["source"] = "IMA同库本地快照：" + item["source"]
                ima_records.append(item)
            fallback_note = "；".join(filter(None, [fallback_note, "话术使用同库 P2 本地快照（表达参考，非参数依据）"]))
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
        if needs_talks:
            records.extend(local_search(product, context_query + " 门店顾客 卖点 话术", 1, talks_only=True))
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
    """每篇本方资料独立匹配，字段不得跨越其他字段，保留原样值与来源句。"""
    details: List[Dict[str, str]] = []
    pending: List[str] = []
    capacity = r"\d+(?:\.\d+)?\s*(?:/\s*\d+(?:\.\d+)?\s*)?(?:GB|TB)(?:\s*/\s*\d+(?:\.\d+)?\s*(?:GB|TB))*"
    prefix = r"\s*(?:为|是|容量为|容量|配置为|配置)?\s*[：:|]?\s*"
    patterns = [
        ("操作系统", r"(?:操作系统|系统)" + prefix + r"(HarmonyOS\s*[\d.]+|Windows\s*11(?:\s*家庭版)?|Linux(?:\s*V?\d+)?)"),
        ("处理器", r"(?:处理器|CPU)" + prefix + r"(麒麟\s*[A-Za-z0-9]+(?:\s*Plus)?|Ultra\s+\d+\s+\d+[A-Za-z]*)"),
        ("重量", r"(?:机身官方重量|重量)" + prefix + r"((?:约\s*)?\d+(?:\.\d+)?\s*(?:kg|g|千克|克))"),
        ("内存", r"(?:内存|\bRAM\b)" + prefix + "(" + capacity.replace("(?:GB|TB)", "GB") + ")"),
        ("存储", r"(?:存储|硬盘|\bSSD\b)" + prefix + "(" + capacity + ")"),
        ("屏幕尺寸", r"(?:屏幕|尺寸)" + prefix + r"(\d+(?:\.\d+)?\s*英寸)"),
    ]
    records = [item for item in evidence.get("records", []) if _is_own_fact(product, item)]
    # 产品事实卡为规范基准，P2 与竞品在入口处即被隔离。
    records.sort(key=lambda item: ("产品事实" not in item.get("title", ""), "P1-" not in item.get("source", "")))
    for item in records:
        raw = _clean_media_text(item.get("highlight", ""))
        if len(raw) < 10 or raw == "目录浏览命中":
            continue
        for label, pattern in patterns:
            for match in re.finditer(pattern, raw, flags=re.I):
                value = match.group(1)  # 不重写空格或单位，核验时可原样回查。
                start = max(raw.rfind("。", 0, match.start()), raw.rfind("；", 0, match.start()), raw.rfind(";", 0, match.start())) + 1
                ends = [pos for sep in ("。", "；", ";") if (pos := raw.find(sep, match.end())) >= 0]
                end = min(ends) + 1 if ends else len(raw)
                sentence = raw[max(start, match.start() - 30):min(end, match.end() + 60)]
                if value not in sentence:
                    pending.append(f"{label}：{value} 无法在本方来源句原样回查，已移除")
                    continue
                details.append({"label": label, "value": value, "sentence": sentence,
                                "title": item.get("title", "资料"), "source": item.get("source", ""), "raw": raw})
    accepted = []
    for label, _ in patterns:
        candidates = [detail for detail in details if detail["label"] == label]
        if not candidates:
            continue
        if label == "重量":
            def grams(value):
                number = float(re.search(r"\d+(?:\.\d+)?", value).group())
                return round(number * (1000 if re.search(r"kg|千克", value, re.I) else 1), 3)
            canonical = candidates[0]
            conflicts = [candidate for candidate in candidates if grams(candidate["value"]) != grams(canonical["value"])]
            for conflict in conflicts:
                pending.append(f"重量冲突：{conflict['value']}（{conflict['title']}；原句：{conflict['sentence']}），与本方规范值 {canonical['value']} 不符，已丢弃")
            # 两份产品事实卡冲突时不擅自选值。
            if any("产品事实" in candidate["title"] for candidate in conflicts) and "产品事实" in canonical["title"]:
                pending.append("本方产品事实卡重量冲突，停止使用重量，待人工确认")
                continue
        accepted.append(candidates[0])
    for item in evidence.get("records", []):
        if _is_competitor(item):
            excluded_weight = re.search(patterns[2][1], _clean_media_text(item.get("highlight", "")), re.I)
            excluded_note = f"；其中重量 {excluded_weight.group(1)} 为非本方参数，未采用" if excluded_weight else ""
            pending.append(f"已隔离竞品对照资料：{item.get('title', '资料')}（仅供参考，不作事实依据）{excluded_note}")
    evidence["fact_details"] = accepted
    evidence["unconfirmed"] = list(dict.fromkeys(pending))
    return [f"{detail['label']}：{'约 ' if detail['label'] == '重量' and not detail['value'].startswith('约') else ''}{detail['value']}" for detail in accepted]


def _recommended_talks(evidence: Dict[str, Any]) -> List[Dict[str, str]]:
    """只取非数字的表达话术；P2 参数不会被带入事实池。"""
    talks = []
    for item in evidence.get("records", []):
        if not _is_talk(item) or _is_competitor(item):
            continue
        raw = _safe_text(item.get("highlight"))
        quoted = re.findall(r'[“「]([^”」]+)[”」]', raw)
        for line in quoted:
            if not re.search(r"\d|全面无短板|最高|第一|唯一", line) and len(line) >= 12:
                talks.append({"text": line, "title": item.get("title", "话术卡"), "source": item.get("source", "")})
                break
    return talks[:2]


def _verify_article(article: str, evidence: Dict[str, Any]) -> bool:
    """拦截模型自行补写的参数与内存/存储错位，失败回退已有事实模板。"""
    body = article.split("配图方案", 1)[0]
    details = evidence.get("fact_details", [])
    normalize = lambda value: re.sub(r"\s+", "", value).lower()
    quantity = r"\d+(?:\.\d+)?\s*(?:GB|TB|kg|g|千克|克|英寸|nits|Wh|W|小时|元)(?![A-Za-z])"
    allowed = {normalize(match.group()) for item in details for match in re.finditer(quantity, item['value'], re.I)}
    warnings = []
    for match in re.finditer(quantity, body, re.I):
        if normalize(match.group()) not in allowed:
            warnings.append(f"正文参数 {match.group()} 无已核验的本方来源，已撤回模型稿并使用事实模板")
    for label, alias, stop in (("内存", r"内存|RAM", r"存储|SSD|硬盘"), ("存储", r"存储|SSD|硬盘", r"内存|RAM")):
        allowed_values = {normalize(match.group()) for item in details if item['label'] == label
                          for match in re.finditer(r"\d+\s*(?:GB|TB)", item['value'], re.I)}
        for match in re.finditer(rf"(?:{alias})([^。；;\n]{{0,60}})", body, re.I):
            phrase = re.split(stop, match.group(1), maxsplit=1, flags=re.I)[0]
            for capacity in re.finditer(r"\d+\s*(?:GB|TB)", phrase, re.I):
                if normalize(capacity.group()) not in allowed_values:
                    warnings.append(f"正文{label}数值 {capacity.group()} 与本方字段不符，已撤回模型稿并使用事实模板")
    evidence["article_warnings"] = list(dict.fromkeys(warnings))
    return not warnings


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
    """把参数翻译成自然口吻的卖点句；修复标签重复 bug（fact 自带"字段："前缀）。"""
    sentences: List[str] = []
    for fact in facts:
        label, _, value = fact.partition("：")
        value = value.strip() or fact
        if label == "重量":
            sentences.append(f"整机重量{value}，天天背去教室/差旅通勤，这个数字就是幸福感本身，到店亲手掂一掂最直观。")
        elif label == "屏幕尺寸":
            sentences.append(f"屏幕{value}，课件、表格、修图的时候，看得舒服比参数好看更重要。")
        elif re.search(r"HarmonyOS|Windows|Linux", value, re.I):
            sentences.append(f"系统是{value}，下单前把最常用的那几个软件逐个装一遍试试，比看任何评测都靠谱。")
        elif label == "处理器":
            sentences.append(f"处理器{value}，这是官方资料里的配置事实——日常多开网页、剪个视频够不够用，建议带自己的素材到店实测。")
        elif label == "内存":
            sentences.append(f"内存{value}，多开软件不卡的前提，按自己的使用强度选就行。")
        elif label == "存储":
            sentences.append(f"存储{value}，四年课件、项目文件和素材往里塞之前，先估一下自己的量。")
        else:
            sentences.append(f"{label}{value}，具体以销售 SKU 为准。")
    return sentences


def offline_article(product: str, audience: str, scene: str, platform: str, requirements: str, evidence: Dict[str, Any]) -> str:
    facts = _extract_facts(product, evidence)
    sentences = _fact_sentences(facts, scene)
    talks = _recommended_talks(evidence)
    talk = talks[0] if talks else None
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
        if talk:
            body += f"\n\n{talk['text']}（话术来源：{talk['title']}）"
    elif platform == "门店朋友圈":
        title = f"{product}：{scene or '开学办公'}"
        core_fact = next((fact for fact in facts if fact.startswith("重量：")), facts[0] if facts else "")
        core_value = core_fact.partition("：")[2].strip() or "配置按实际 SKU 核对"
        hook = talk['text'] if talk else "带上常用软件和外设，现场逐项验证。"
        body = f"{scene or '到店选购'}看这台：{core_value}，{hook}\n到店体验，私信咨询。"
        if talk:
            body += f"\n（话术来源：{talk['title']}）"
    elif platform == "小红书":
        audience_word = (audience or "宝子们").strip()
        title = f"{scene or '开学季'}买电脑前，先把这几个数字看明白🎓"
        detail = "\n\n".join(sentences) if sentences else "先想清楚每天怎么用、装什么软件、背不背出门，再谈买哪款。"
        talk_line = f"\n\n{talk['text']}\n" if talk else ""
        body = (
            f"{audience_word}看过来👀 {scene or '开学季'}挑电脑最容易踩的坑，就是只看颜值不核对配置。\n\n"
            f"{detail}\n{talk_line}\n"
            f"参数都来自华为官方资料，放心抄作业📝 建议带上自己最常用的软件和文件到店跑一遍，合不合适上手就知道。\n\n"
            f"评论区聊聊你的使用场景，帮你看配置～{tags}"
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
    return f"### 标题\n{title}\n\n### 导语与正文\n{body}\n\n### 补充要求响应\n{requirements or '按场景化、事实优先原则生成。'}\n\n### 配图方案\n{_image_plan(evidence.get('images', []), platform)}\n\n### 平台标签\n{platform}{tags}"


def call_llm(product: str, audience: str, scene: str, platform: str, requirements: str, evidence: Dict[str, Any],
             variant: str = "", attachment_text: str = "") -> str:
    """优先调用魔搭，失败后尝试 DeepSeek，全部失败交给离线生成器。软文高温度、核验低温度分离。"""
    _extract_facts(product, evidence)
    # 模型仅收到逐条核验的事实，不接收竞品原文或 P2 中的参数。
    records = evidence.get("fact_details", [])
    evidence_text = "\n".join(
        f"- {item['label']}：{item['value']} | 来源：{item['title']} | 本方来源句：{item['sentence']}"
        for item in records
    ) or "- 当前无在线资料，必须只使用输入与离线命中的事实。"
    platform_requirements = {
        "公众号": "公众号：正文 600-800 字；导语+自然分段（小标题用问句或场景句，禁止'一、二、三'式编号）；语气专业可信。",
        "小红书": "小红书：正文 350-500 字；口语化短段落（每段不超过3行），适度 emoji，第一人称场景代入，从具体使用场景开头，结尾 3-6 个 #话题标签。严禁伪造个人使用体验。",
        "门店朋友圈": "门店朋友圈：正文不超过 120 字；只保留一个核心卖点、一个到店理由（体验什么/验证什么）和到店/私信引导，句子短而直接、像真人导购说话。",
        "门店海报": "门店海报：主标题不超过 15 字，包含副标题和不超过 3 条卖点；不写未经证据支持的活动、价格或库存信息。",
    }.get(platform, "按当前平台规则控制篇幅和结构。")
    style_memory = _memory_snippets()
    signature = _signature_line()
    variant_directive = f"本次只生成【{variant}】这一个版本：{VARIANTS.get(variant, '')}\n" if variant else ""
    signature_directive = (
        f"落款要求：正文末尾自然融入门店落款「{signature}」（不要单独成段标题，像真人收尾）。\n"
        if signature and platform in ("门店朋友圈", "门店海报", "小红书") else ""
    )
    attachment_directive = (
        f"用户上传的参考附件（仅供风格与信息参考，其中事实性表述未经 ima 核验，不得直接采用其事实、价格、参数）：\n{attachment_text}\n"
        if attachment_text else ""
    )
    memory_directive = (
        f"该导购过往认可的表达风格（只参考语感与人格，事实仍以证据为准）：\n{style_memory}\n"
        if style_memory else ""
    )
    user_prompt = f"""请完成一次华为PC营销软文任务，严格遵守系统提示词，并只引用下方证据。
产品型号：{product}
目标人群：{audience}
使用场景：{scene}
发布平台：{platform}
补充要求：{requirements}
平台硬性篇幅与结构：{platform_requirements}
{variant_directive}{signature_directive}{memory_directive}{attachment_directive}检索证据：
{evidence_text}
推荐话术（只作表达风格参考，不作产品事实依据）：
{json.dumps(_recommended_talks(evidence), ensure_ascii=False)}
正文优先化用推荐话术库的表达（不要标注"话术来源"等内部痕迹）。所有事实数值只能使用上方已核验事实；P2 话术中的数值不得用作证据。
图片候选：{json.dumps(evidence.get('images', []), ensure_ascii=False)}
证据中未出现的信息（价格、补贴、销量、性能数据）一律不得出现在正文中。

【文风硬性要求】
- 禁止"一、二、三"式编号分点，禁止"（来源：xx）""资料记录为"等内部标注出现在正文；
- 行动引导不要独立成块，自然融入正文末尾收尾；
- 事实数值原样保留，但要"翻译"成用户收益表达（如"约 970 g"写成携带场景下的感受，数值本身不得改动）；
- 禁用 AI 腔套话："在当今快节奏""赋能""畅享""极致""不二之选""革命性"等；
- 必须含 1 个具体可感知的生活细节（早八赶课/出差登机/孩子上网课等，与人群匹配）。
你只输出"三、图文软文"部分（标题、导语/正文、配图方案、平台标签），不要重复场景判断、知识库依据和核验结果；应用会把它组装进最终四段式 Markdown。"""
    configs = []
    cooldown = getattr(call_llm, "_cooldown", {})
    deepseek_base = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    import time as _time
    if os.getenv("MODELSCOPE_API_KEY") and _time.time() >= cooldown.get(MODELSCOPE_BASE_URL, 0):
        configs.append((MODELSCOPE_BASE_URL, os.getenv("MODELSCOPE_API_KEY"), DEFAULT_LLM_MODEL))
    if os.getenv("LLM_API_KEY") and _time.time() >= cooldown.get(deepseek_base, 0):
        configs.append((deepseek_base, os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL", "deepseek-chat")))
    if not configs:
        return ""
    try:
        from openai import OpenAI
    except Exception:
        return ""
    for base_url, api_key, model in configs:
        try:
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=45, max_retries=0)
            max_tokens = 2200 if platform in ("公众号",) else 1400
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
                temperature=0.9,
                max_tokens=max_tokens,
                stream=True,
            )
            chunks = []
            for piece in stream:
                delta = piece.choices[0].delta if piece.choices else None
                if delta and delta.content:
                    chunks.append(delta.content)
            content = "".join(chunks)
            if content:
                # 模型偶尔仍会返回完整四段，这里只取第三段，避免页面重复。
                if "三、图文软文" in content:
                    content = content.split("三、图文软文", 1)[1]
                if "四、核验结果" in content:
                    content = content.split("四、核验结果", 1)[0]
                content = content.strip().lstrip("：:").strip()
                if len(_clean_media_text(content)) >= 80:
                    call_llm._cooldown = {k: v for k, v in getattr(call_llm, "_cooldown", {}).items() if k != base_url}
                    return content
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            if status_code in (401, 403):
                # 凭证类恒定失败：熔断10分钟，避免每次生成都白等一次 TLS 往返。
                cooldown = getattr(call_llm, "_cooldown", {})
                import time as _time
                cooldown[base_url] = _time.time() + 600
                call_llm._cooldown = cooldown
                print(f"[LLM熔断] {base_url} 返回 {status_code}，10分钟内跳过该通道")
            continue
    return ""


def _presentation_url(value: Any) -> str:
    """展示层只接受真实 HTTP(S) 地址，防止资料字段注入脚本。"""
    url = _safe_text(value)
    try:
        parsed = urlsplit(url)
        return url if parsed.scheme in ("http", "https") and parsed.netloc else ""
    except ValueError:
        return ""


def _excerpt_fragments(text: str, facts: List[str]) -> List[str]:
    """只选择原文中的相关短句/字段，不参与事实提取或核验。"""
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"!?\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    fragments = re.split(r"[\r\n。！？；]+|#{1,6}\s*|\s[-*]\s|\|", text)
    candidates = []
    for fragment in fragments:
        fragment = re.sub(r"^[\s>*#\-]+|[*`]+", "", fragment).strip()
        if len(fragment) >= 10 and fragment != "目录浏览命中" and fragment not in candidates:
            candidates.append(fragment)
    # 参数值优先，其次属性词；每条资料最多三个原文字段。
    values = [part.strip() for fact in facts for part in fact.split("：")[1:] if part.strip()]
    attributes = [fact.split("：", 1)[0] for fact in facts]
    keywords = list(dict.fromkeys(values + attributes))
    normalize = lambda value: re.sub(r"\s+", "", value).lower()
    def score(fragment):
        normalized = normalize(fragment)
        return sum((3 if keyword in values else 1) for keyword in keywords if normalize(keyword) in normalized)
    ranked = sorted(enumerate(candidates), key=lambda pair: (-score(pair[1]), pair[0]))
    chosen = sorted(index for index, fragment in ranked[:3] if score(fragment))
    if not chosen and candidates:
        chosen = [0]
    result = []
    for index in chosen:
        fragment = candidates[index]
        if len(fragment) > 160:
            # 长字段截取到命中点附近，避免把整段资料再次贴到页面。
            matches = [fragment.lower().find(keyword.lower()) for keyword in keywords]
            matches = [position for position in matches if position >= 0]
            start = max(0, min(matches) - 35) if matches else 0
            fragment = ("…" if start else "") + fragment[start:start + 160] + "…"
        result.append(fragment)
    return result


def _mirror_image_path(image: Dict[str, Any]) -> Path | None:
    """按已有图片索引定位镜像；仅展示层读取文件，不改变检索结果。"""
    for row in local_image_index():
        if row.get("文件名") != image.get("name"):
            continue
        if image.get("url") and row.get("原始图片URL") != image["url"]:
            continue
        folder = KB_ROOT / Path(row["_file"]).parent
        for candidate in (folder / "images" / row["文件名"], folder / row["文件名"]):
            path = candidate.resolve()
            if path.is_relative_to(KB_ROOT.resolve()) and path.is_file():
                return path
    return None


def _render_basis(product: str, evidence: Dict[str, Any], facts: List[str]) -> str:
    """依据与图片专用 HTML；所有资料文字和地址先转义。"""
    escape = html_lib.escape
    fact_cards = []
    for fact in facts:
        label, separator, value = fact.partition("：")
        fact_cards.append(
            '<div class="fact-stat">'
            f'<span class="fact-label">{escape(label if separator else "事实点")}</span>'
            f'<strong class="fact-value">{escape(value if separator else fact)}</strong>'
            '</div>'
        )
    cards = []
    talk_cards = []
    for item in evidence.get("records", []):
        excerpts = _excerpt_fragments(_safe_text(item.get("highlight")), facts)
        marked = "<br>".join(
            f'<mark class="evidence-mark">{escape(fragment)}</mark>'
            for fragment in excerpts
        ) or "暂无可展示的原文摘录，待确认。"
        url_info = item.get("url_info") or {}
        url = next((url for value in (item.get("url"), url_info.get("url") if isinstance(url_info, dict) else "", item.get("source"))
                    if (url := _presentation_url(value))), "")
        link = (f'<p><a href="{escape(url, quote=True)}" target="_blank" rel="noopener">📄 查看知识库原文快照</a></p>' if url else "")
        source = _safe_text(item.get("source"))
        competitor = _is_competitor(item)
        talk = _is_talk(item) and not competitor
        if competitor:
            source += "｜竞品对照资料（仅供参考，不作事实依据）"
        elif talk:
            source += "｜P2 卖点话术（表达参考，事实数值以 P1/事实卡为准）"
        title = escape(_safe_text(item.get("title")) or "资料")
        if talk:
            talk_cards.append(
                '<blockquote class="talk-quote">'
                f'<div class="talk-kicker">推荐话术</div><p>{marked}</p>'
                f'<footer>来源：P2-卖点话术库 · {title}</footer>{link}</blockquote>'
            )
        else:
            badge = '<span class="reference-badge">仅供参考</span>' if competitor else ""
            card_class = "evidence-card competitor-card" if competitor else "evidence-card"
            cards.append(
                f'<article class="{card_class}">{badge}'
                f'<div class="evidence-title"><strong>{title}</strong></div>'
                f'<div class="evidence-source">来源：{escape(source)}</div>'
                f'<div class="evidence-excerpt"><span>关键摘录</span>{marked}</div>{link}</article>'
            )
    pictures = []
    for image in evidence.get("images", [])[:6]:
        mirror = "离线" in evidence.get("mode", "") or "本地镜像" in _safe_text(image.get("source"))
        src = ""
        note = ""
        if mirror:
            path = _mirror_image_path(image)
            if path:
                try:
                    if path.stat().st_size >= 2 * 1024 * 1024:
                        note = "图片过大未内嵌，见本地镜像目录"
                    else:
                        mime = mimetypes.guess_type(path.name)[0]
                        if mime in ("image/jpeg", "image/png", "image/webp", "image/gif", "image/avif"):
                            src = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
                        else:
                            note = "图片格式不支持内嵌，见本地镜像目录"
                except OSError:
                    note = "本地镜像暂不可读，见本地镜像目录"
            else:
                note = "本地镜像文件缺失，待确认"
        else:
            src = _presentation_url(image.get("url"))
            if not src:
                note = "暂无可用图片地址，待确认"
        name = escape(_safe_text(image.get("name")) or "图片")
        source = "ima知识库本地镜像" if mirror else "ima 知识库"
        picture = (f'<img src="{escape(src, quote=True)}" alt="{name}">' if src else f'<p>{note}</p>')
        url = _presentation_url(image.get("url"))
        source_link = f'<br><a href="{escape(url, quote=True)}" target="_blank" rel="noopener">图片原始来源</a>' if url else ""
        pictures.append(f'<figure>{picture}<figcaption><strong>{name}</strong><br>来源：{source}{source_link}</figcaption></figure>')
    return (
        f'<div class="basis-heading"><span>本方产品依据</span><strong>{escape(product)}</strong></div>'
        + ('<div class="fact-grid">' + "".join(fact_cards) + '</div>' if fact_cards else '<p class="empty-note">当前未提取到结构化事实，发布前需人工确认。</p>')
        + ('<section class="talk-group"><h4>卖点话术</h4>' + "".join(talk_cards) + '</section>' if talk_cards else "")
        + '<section class="evidence-group"><h4>资料摘录与溯源</h4><div class="evidence-list">'
        + ("".join(cards) or '<p class="empty-note">未命中资料，待确认。</p>') + '</div></section>'
        + '<section class="image-group"><h4>图片来源清单</h4><p class="image-help">原图已内嵌，可右键另存为。</p><div class="image-grid">'
        + ("".join(pictures) or '<p class="empty-note">暂无可展示的图片。</p>') + "</div></section>"
        + _store_images(product)
    )


# ═══════════════════ 个人中心：本地数据层（user_data/，不上传 GitHub） ═══════════════════
USER_DATA = ROOT / "user_data"
ASSET_ROOT = USER_DATA / "assets"

VARIANTS = {
    "场景故事型": "从目标人群的一天/一个具体场景切入，讲故事感，先场景后产品，情绪共鸣优先。",
    "干货清单型": "以实用选购要点/核对清单切入，节奏快、信息密度高，但禁止编号分点，用短段落自然衔接。",
    "体验邀约型": "以门店体验邀约切入，强调到店亲手验证什么、体验顾问能提供什么帮助，转化导向。",
}
VARIANT_NAMES = list(VARIANTS.keys())


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_profile() -> Dict[str, Any]:
    return _load_json(USER_DATA / "profile.json", {})


def save_profile(data: Dict[str, Any]) -> str:
    USER_DATA.mkdir(exist_ok=True)
    (USER_DATA / "profile.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return "✅ 个人资料已保存，下次生成自动生效。"


def _signature_line() -> str:
    profile = load_profile()
    if not profile.get("auto_signature", True):
        return ""
    custom = _safe_text(profile.get("sign_template"))
    if custom:
        return custom
    name = _safe_text(profile.get("name"))
    store = _safe_text(profile.get("store_name"))
    store_type = _safe_text(profile.get("store_type"))
    if not (name or store):
        return ""
    parts = [p for p in (store_type + store if store else "", f"体验顾问 {name}" if name else "") if p]
    line = " · ".join(parts)
    wecom = _safe_text(profile.get("wecom"))
    if wecom:
        line += f"（企业微信：{wecom}）"
    return line


def _memory_snippets() -> str:
    if not load_profile().get("memory_on", True):
        return ""
    snippets = []
    for path in (USER_DATA / "favorites.jsonl",):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines()[-3:]:
            try:
                record = json.loads(line)
                text = _clean_media_text(record.get("article", ""))[:260]
                if text:
                    snippets.append(f"- （{record.get('platform', '')}）{text}")
            except Exception:
                continue
    return "\n".join(snippets)


def _append_jsonl(path: Path, record: Dict[str, Any]):
    USER_DATA.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_history(product: str, platform: str, scene: str, versions: List[str]):
    _append_jsonl(USER_DATA / "history.jsonl", {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "product": product, "platform": platform, "scene": scene,
        "versions": versions, "article": versions[0] if versions else "",
    })


def save_favorite(index: int, state: Dict[str, Any]) -> str:
    versions = (state or {}).get("versions", [])
    if index >= len(versions) or not versions[index]:
        return "该版本暂无内容，请先成功生成。"
    _append_jsonl(USER_DATA / "favorites.jsonl", {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "product": state.get("product", ""), "platform": state.get("platform", ""),
        "variant": VARIANT_NAMES[index] if index < len(VARIANT_NAMES) else f"版本{index + 1}",
        "article": versions[index],
    })
    return f"⭐ 已收藏「{VARIANT_NAMES[index]}」，下次生成将参考它的语感。"


def _records_html(path: Path, kind: str) -> str:
    if not path.exists():
        return '<p class="empty-note">暂无记录，生成内容后会自动出现在这里。</p>'
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in reversed(lines[-12:]):
        try:
            record = json.loads(line)
        except Exception:
            continue
        if kind == "history":
            preview = _clean_media_text((record.get("versions") or [record.get("article", "")])[0])[:90]
            rows.append(
                f'<details class="record-item"><summary><strong>{escape_h(record.get("ts", ""))}</strong>'
                f'　{escape_h(record.get("product", ""))} · {escape_h(record.get("platform", ""))}'
                f'<span class="record-preview">{escape_h(preview)}…</span></summary>'
                f'<div class="record-body">{escape_h((record.get("versions") or [""])[0])}</div></details>'
            )
        else:
            preview = _clean_media_text(record.get("article", ""))[:120]
            rows.append(
                f'<div class="record-item fav"><div class="fav-meta">⭐ {escape_h(record.get("ts", ""))}　'
                f'{escape_h(record.get("variant", ""))} · {escape_h(record.get("product", ""))} · {escape_h(record.get("platform", ""))}</div>'
                f'<div class="record-body">{escape_h(preview)}…</div></div>'
            )
    return "".join(rows) or '<p class="empty-note">暂无记录。</p>'


def escape_h(value: Any) -> str:
    return html_lib.escape(_safe_text(value))


def upload_store_image(model: str, file_path: str, note: str) -> Tuple[str, str]:
    """门店实拍入库：强制型号核验（型号矩阵同源）+ 解码检查 + MD5 去重。"""
    import hashlib
    model = _safe_text(model)
    if model not in load_model_choices():
        return "❌ 核验失败：型号不在 P1 全型号参数联合矩阵中，请核对后重试。", gr.skip()
    if not file_path:
        return "❌ 请先选择要上传的图片文件。", gr.skip()
    source = Path(file_path)
    if source.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        return "❌ 仅支持 jpg / png / webp 图片。", gr.skip()
    try:
        data = source.read_bytes()
    except OSError:
        return "❌ 文件读取失败，请重试。", gr.skip()
    if len(data) > 10 * 1024 * 1024:
        return "❌ 图片超过 10MB，请压缩后上传。", gr.skip()
    digest = hashlib.md5(data).hexdigest()
    target_dir = ASSET_ROOT / model
    target_dir.mkdir(parents=True, exist_ok=True)
    meta_path = ASSET_ROOT / "assets.jsonl"
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("md5") == digest:
                    return "⚠️ 这张图片已在素材库中（MD5 去重），无需重复上传。", gallery_html()
            except Exception:
                continue
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{digest[:8]}{source.suffix.lower()}"
    (target_dir / filename).write_bytes(data)
    _append_jsonl(meta_path, {"file": f"{model}/{filename}", "model": model, "md5": digest,
                              "note": _safe_text(note), "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    return f"✅ 核验通过，已入库：{model} / {filename}", gallery_html()


def gallery_html() -> str:
    meta_path = ASSET_ROOT / "assets.jsonl"
    if not meta_path.exists():
        return '<p class="empty-note">素材库为空。上传门店实物照片并绑定型号后，生成软文时会自动出现在"本店实拍"分组。</p>'
    figures = []
    for line in reversed(meta_path.read_text(encoding="utf-8").splitlines()[-18:]):
        try:
            record = json.loads(line)
        except Exception:
            continue
        path = ASSET_ROOT / record.get("file", "")
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            src = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
        except OSError:
            continue
        figures.append(
            f'<figure><img src="{src}" alt="{escape_h(record.get("model"))}">'
            f'<figcaption><strong>{escape_h(record.get("model"))}</strong><br>'
            f'{escape_h(record.get("ts", ""))} {escape_h(record.get("note", ""))}</figcaption></figure>'
        )
    return ('<div class="image-grid">' + "".join(figures) + "</div>") if figures else '<p class="empty-note">暂无可展示的素材。</p>'


def _store_images(product: str) -> str:
    """生成时把同型号门店实拍追加为独立分组，与 ima 图片严格分开标注。"""
    meta_path = ASSET_ROOT / "assets.jsonl"
    if not meta_path.exists():
        return ""
    figures = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except Exception:
            continue
        if record.get("model") != product:
            continue
        path = ASSET_ROOT / record.get("file", "")
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            src = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
        except OSError:
            continue
        figures.append(f'<figure><img src="{src}" alt="本店实拍"><figcaption><strong>本店实拍</strong><br>来源：门店实拍素材库（{escape_h(record.get("ts", ""))}）</figcaption></figure>')
        if len(figures) >= 3:
            break
    if not figures:
        return ""
    return '<section class="image-group"><h4>本店实拍（门店素材库 · 展示补充，评分口径图片以 ima 为准）</h4><div class="image-grid">' + "".join(figures) + "</div></section>"


def _read_attachment(file_path: str) -> str:
    if not file_path:
        return ""
    try:
        raw = Path(file_path).read_bytes()[:2 * 1024 * 1024]
    except OSError:
        return ""
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(encoding)[:3000]
        except UnicodeDecodeError:
            continue
    return ""


def _honor_loading_html(stage: str = "retrieve", done_count: int = 0) -> str:
    """等待动画（王者荣耀加载界面风格）：深蓝金框、旋转光圈、文案轮换、四步进度点亮。纯 CSS 实现，无 JS 依赖。"""
    if stage == "retrieve":
        lit = 1
        lines = ["⚔ 正在唤醒 ima 知识库…", "📜 翻检 1519 项官方共享资料…",
                 "🔍 定位型号事实卡与原文…", "🧭 好内容值得等待，荣耀加载中…"]
    else:
        lit = 3
        lines = ["✍️ AI 导购正在打磨三版文案…", f"📦 已完成 {done_count}/3 版，先完成先上屏…",
                 "🛡 所有数值将逐条回查知识库原句…", "🚫 合规红线扫描：拒绝绝对化与虚构价格…"]
    line_html = "".join(f'<p class="honor-line">{t}</p>' for t in lines)
    step_names = ("① 检索", "② 事实", "③ 生成", "④ 核验")
    step_html = "".join(
        f'<div class="honor-step{" lit" if i < lit else ""}">{name}</div>'
        for i, name in enumerate(step_names)
    )
    css = """<style>
.honor-load{position:relative;border-radius:14px;border:1px solid rgba(201,169,106,.36);
  background:linear-gradient(160deg,#0B1437,#16224A 60%,#1B2A4A);padding:20px 24px;overflow:hidden;min-height:150px}
.honor-ring{position:absolute;right:-44px;top:-44px;width:170px;height:170px;border-radius:50%;
  background:conic-gradient(from 0deg,rgba(201,169,106,0),rgba(201,169,106,.55),rgba(201,169,106,0));
  animation:honorSpin 2.4s linear infinite;opacity:.5;filter:blur(2px)}
@keyframes honorSpin{to{transform:rotate(360deg)}}
.honor-lines{position:relative;height:64px;margin:4px 0 14px}
.honor-line{position:absolute;left:0;top:0;right:0;display:flex;align-items:center;height:64px;margin:0;
  font-size:17px;font-weight:600;color:#F5F0E6;opacity:0;animation:honorLine 12s linear infinite;
  text-shadow:0 0 18px rgba(201,169,106,.35)}
.honor-line:nth-child(1){animation-delay:0s}.honor-line:nth-child(2){animation-delay:3s}
.honor-line:nth-child(3){animation-delay:6s}.honor-line:nth-child(4){animation-delay:9s}
@keyframes honorLine{0%{opacity:0;transform:translateY(10px)}4%{opacity:1;transform:translateY(0)}
  27%{opacity:1}32%{opacity:0;transform:translateY(-10px)}100%{opacity:0}}
.honor-steps{display:flex;gap:8px}
.honor-step{flex:1;text-align:center;padding:8px 0;font-size:12px;color:rgba(245,240,230,.4);
  border:1px solid rgba(245,240,230,.14);border-radius:8px;background:rgba(245,240,230,.04)}
.honor-step.lit{color:#FFE3A7;border-color:rgba(201,169,106,.55);background:rgba(201,169,106,.10);
  animation:honorPulse 1.8s ease-in-out infinite}
@keyframes honorPulse{0%,100%{box-shadow:0 0 6px rgba(201,169,106,.15)}50%{box-shadow:0 0 16px rgba(201,169,106,.4)}}
.honor-foot{margin-top:10px;font-size:11px;color:rgba(245,240,230,.45);text-align:center}
</style>"""
    return (css
            + '<div class="honor-load"><div class="honor-ring"></div>'
            + f'<div class="honor-lines">{line_html}</div>'
            + f'<div class="honor-steps">{step_html}</div>'
            + '<div class="honor-foot">所有数值均来自 ima 知识库原文 · 逐条可溯源 · 无依据即拒绝</div></div>')


def _publish_text(article: str) -> str:
    """把成稿净化为可直接粘贴到小红书/朋友圈的纯文本。"""
    text = _safe_text(article)
    text = re.split(r"###\s*(补充要求响应|配图方案|平台标签|行动引导)", text)[0]
    text = re.sub(r"^#{1,6}\s*[^\n]*$", "", text, flags=re.M)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"^[-*]\s+", "", text, flags=re.M)
    text = re.sub(r"^>\s*⏱[^\n]*$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def export_publish_txt(versions_state: Dict[str, Any], platform: str):
    versions = (versions_state or {}).get("versions", [])
    parts = []
    for name, article in zip(VARIANT_NAMES, versions):
        cleaned = _publish_text(article)
        if cleaned:
            parts.append(f"=====【{name}】{platform}发布版 =====\n\n{cleaned}")
    content = "\n\n\n".join(parts) or "尚未生成内容，请先点击生成。"
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix=f"{platform}_发布版_{stamp}_", encoding="utf-8-sig", mode="w")
    handle.write(content)
    handle.close()
    return handle.name


def generate_flow(product: str, audience: str, scene: str, platform: str, requirements: str, revision: str = "",
                  attachment_text: str = ""):
    product = _safe_text(product) or load_model_choices()[0]
    audience = _safe_text(audience)
    scene = _safe_text(scene)
    platform = _safe_text(platform) or "公众号"
    requirements = "；".join(x for x in (_safe_text(requirements), _safe_text(revision)) if x)
    evidence = retrieve_evidence(product, audience, scene, platform, requirements)
    facts = _extract_facts(product, evidence)
    # 三版本并列生成：同一套核验事实，三种切入人格；并行调用把在线延迟压到单版水平。
    # 超时治理：LLM 客户端 25s 快速失败（max_retries=0），三版并发互不阻塞，全部失败则离线单版本。
    def _gen_one(variant_name):
        llm_text = call_llm(product, audience, scene, platform, requirements, evidence,
                            variant=variant_name, attachment_text=attachment_text)
        if llm_text and not _verify_article(llm_text, evidence):
            return ""
        return llm_text or ""

    with ThreadPoolExecutor(max_workers=3) as pool:
        versions = [text for text in pool.map(_gen_one, VARIANT_NAMES) if text]
    if not versions:
        versions = [offline_article(product, audience, scene, platform, requirements, evidence)]
    signature = _signature_line()
    if signature and platform in ("门店朋友圈", "门店海报", "小红书"):
        versions = [version + f"\n\n{signature}" for version in versions]
    article = versions[0]
    versions = (versions + ["", ""])[:3]
    save_history(product, platform, scene, [v for v in versions if v])
    facts = _extract_facts(product, evidence)
    # 最后逐条回查本方原句；不使用“命中型号”代替数值证据核验。
    trace_lines = []
    for detail in evidence.get("fact_details", []):
        if detail["value"] in detail["sentence"] and detail["sentence"] in detail["raw"]:
            trace_lines.append(f"- ✅ {detail['label']}：{detail['value']}（来源：{detail['title']}；{detail['source']}）\n  > 本方来源句：{detail['sentence']}")
    pending = list(dict.fromkeys(evidence.get("unconfirmed", []) + evidence.get("article_warnings", [])))
    product_key = product.lower()
    fact_records_match = any(
        product_key in " ".join(_safe_text(item.get(key)) for key in ("title", "source", "highlight")).lower()
        for item in evidence.get("records", [])
    )
    image_records_match = bool(evidence.get("images")) and all(
        product_key in " ".join(_safe_text(image.get(key)) for key in ("name", "source")).lower()
        for image in evidence.get("images", [])
    )
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
    basis_md = _render_basis(product, evidence, facts)
    verification_md = f"""### 四、核验结果
- 产品事实：{'已绑定包含该型号标识的本次检索资料' if facts and fact_records_match else '待人工确认，未命中该型号的足够事实'}
- 图片型号：{'已按产品型号精确筛选并记录来源' if image_records_match else ('已记录来源，但型号需人工核对' if evidence.get('images') else '未找到精确型号图片，禁止使用外部占位图')}
- 合规检查：已排查绝对化宣传、虚假承诺、无依据对比、价格误导、夸大功能；价格/补贴/库存未写入。
- 待确认信息：具体销售 SKU、系统/内存/存储组合、实际库存与活动政策；发布前以门店和官方页面为准。{'；'.join(pending)}
- 资料模式：{evidence['mode']}"""
    verification_md += "\n\n#### 产品事实逐条回查\n" + ("\n".join(trace_lines) or "未取得可回查的本方事实，不输出数字参数。")
    combined = f"{scene_md}\n\n{basis_md}\n\n### 三、图文软文\n{article}\n\n{verification_md}"
    # 依据区图片是 base64 内嵌的，复制/导出时替换为占位说明，避免满屏乱码。
    combined = re.sub(r"data:image/[^\"')\s]+", "（图片已内嵌在上方依据区，可右键另存）", combined)
    return status, basis_md, scene_md, basis_md, versions, verification_md, combined


def generate_flow_stream(product: str, audience: str, scene: str, platform: str, requirements: str, revision: str = "",
                         attachment_text: str = ""):
    """流式版生成：检索完先把依据区亮出来；三版并行、谁先写完谁先上屏；最后收尾核验。
    每次 yield 与 generate_flow 返回值同构，供 UI 逐步刷新。"""
    import time as _time
    t0 = _time.time()
    product = _safe_text(product) or load_model_choices()[0]
    audience = _safe_text(audience)
    scene = _safe_text(scene)
    platform = _safe_text(platform) or "公众号"
    requirements = "；".join(x for x in (_safe_text(requirements), _safe_text(revision)) if x)
    evidence = retrieve_evidence(product, audience, scene, platform, requirements)
    facts = _extract_facts(product, evidence)
    t_retrieval = _time.time()
    retrieval_sec = t_retrieval - t0
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
    basis_md = _render_basis(product, evidence, facts)
    timing_bar = ('<div style="margin:0 0 8px;padding:6px 12px;border-radius:8px;'
                  'background:rgba(201,169,106,.12);border:1px solid rgba(201,169,106,.36);'
                  f'color:#C9A96A;font-size:12px;">⏱ ima 检索+依据渲染耗时 {retrieval_sec:.1f}s ｜ 三版生成中…</div>')
    basis_md = timing_bar + basis_md
    status += f"\n\n> ⏱ 本次 ima 检索耗时 {retrieval_sec:.1f}s"
    scene_md = f"> ⏱ 场景判断 · 解析耗时 {retrieval_sec:.1f}s（与检索同步完成）\n\n{scene_md}"
    waiting = [f"⏳ 版本生成中（已等待 0s，先完成的先显示）…" for _ in VARIANT_NAMES]
    verification_wait = f"### 四、核验结果\n三版本写作中（已耗时 {retrieval_sec:.1f}s）；完成后逐条回查事实与合规。"
    yield status, basis_md, scene_md, basis_md, list(waiting), verification_wait, f"{scene_md}\n\n{basis_md}"

    def _gen_one(variant_name):
        llm_text = call_llm(product, audience, scene, platform, requirements, evidence,
                            variant=variant_name, attachment_text=attachment_text)
        if llm_text and not _verify_article(llm_text, evidence):
            return ""
        # 归一化模型偶尔打乱的加粗符号（如“**标题：**”），保证版面干净。
        if llm_text:
            llm_text = llm_text.replace("**标题：**", "**标题**：").replace("**标题:**", "**标题**：")
        return llm_text or ""

    done = {name: "" for name in VARIANT_NAMES}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {name: pool.submit(_gen_one, name) for name in VARIANT_NAMES}
        remaining = set(VARIANT_NAMES)
        def _elapsed():
            return _time.time() - t_retrieval
        while remaining:
            _time.sleep(1.0)
            for name in list(remaining):
                if futures[name].done():
                    remaining.discard(name)
                    result = futures[name].result()
                    done[name] = f"> ⏱ {name} · 本版生成耗时 {_elapsed():.1f}s\n\n{result}" if result else ""
                    progress = [done[n] or f"⏳ {n} · 生成中（已等待 {_elapsed():.0f}s）…" for n in VARIANT_NAMES]
                    finished = [done[n] for n in VARIANT_NAMES if done[n]]
                    yield (status, basis_md, scene_md, basis_md, progress,
                           f"### 四、核验结果\n已出 {len(finished)}/3 版（最新一版耗时 {_elapsed():.1f}s），其余写作中。",
                           f"{scene_md}\n\n{basis_md}\n\n### 三、图文软文\n" + "\n\n---\n\n".join(finished))
    versions = [done[name] for name in VARIANT_NAMES if done[name]]
    if not versions:
        versions = [offline_article(product, audience, scene, platform, requirements, evidence)]
    signature = _signature_line()
    if signature and platform in ("门店朋友圈", "门店海报", "小红书"):
        versions = [version + f"\n\n{signature}" for version in versions]
    article = versions[0]
    versions = (versions + ["", ""])[:3]
    save_history(product, platform, scene, [v for v in versions if v])
    facts = _extract_facts(product, evidence)
    trace_lines = []
    for detail in evidence.get("fact_details", []):
        if detail["value"] in detail["sentence"] and detail["sentence"] in detail["raw"]:
            trace_lines.append(f"- ✅ {detail['label']}：{detail['value']}（来源：{detail['title']}；{detail['source']}）\n  > 本方来源句：{detail['sentence']}")
    pending = list(dict.fromkeys(evidence.get("unconfirmed", []) + evidence.get("article_warnings", [])))
    product_key = product.lower()
    fact_records_match = any(
        product_key in " ".join(_safe_text(item.get(key)) for key in ("title", "source", "highlight")).lower()
        for item in evidence.get("records", [])
    )
    image_records_match = bool(evidence.get("images")) and all(
        product_key in " ".join(_safe_text(image.get(key)) for key in ("name", "source")).lower()
        for image in evidence.get("images", [])
    )
    verification_md = f"""### 四、核验结果
- 产品事实：{'已绑定包含该型号标识的本次检索资料' if facts and fact_records_match else '待人工确认，未命中该型号的足够事实'}
- 图片型号：{'已按产品型号精确筛选并记录来源' if image_records_match else ('已记录来源，但型号需人工核对' if evidence.get('images') else '未找到精确型号图片，禁止使用外部占位图')}
- 合规检查：已排查绝对化宣传、虚假承诺、无依据对比、价格误导、夸大功能；价格/补贴/库存未写入。
- 待确认信息：具体销售 SKU、系统/内存/存储组合、实际库存与活动政策；发布前以门店和官方页面为准。{'；'.join(pending)}
- 资料模式：{evidence['mode']}"""
    verification_md += "\n\n#### 产品事实逐条回查\n" + ("\n".join(trace_lines) or "未取得可回查的本方事实，不输出数字参数。")
    combined = f"{scene_md}\n\n{basis_md}\n\n### 三、图文软文\n{article}\n\n{verification_md}"
    combined = re.sub(r"data:image/[^\"')\s]+", "（图片已内嵌在上方依据区，可右键另存）", combined)
    total_sec = _time.time() - t0
    verify_sec = max(total_sec - _elapsed(), 0.0)
    verification_md = (f"> ⏱ 核验+回查耗时 {verify_sec:.1f}s ｜ 全流程累计 {total_sec:.1f}s"
                       f"（检索 {retrieval_sec:.1f}s + 三版生成 {_elapsed():.1f}s + 核验 {verify_sec:.1f}s）\n\n{verification_md}")
    combined = f"{scene_md}\n\n{basis_md}\n\n### 三、图文软文\n{article}\n\n{verification_md}"
    combined = re.sub(r"data:image/[^\"')\s]+", "（图片已内嵌在上方依据区，可右键另存）", combined)
    yield status, basis_md, scene_md, basis_md, versions, verification_md, combined


def export_markdown(content: str):
    content = _safe_text(content)
    if not content:
        content = "尚未生成内容，请先点击生成。"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".md", prefix="华为PC营销软文_", encoding="utf-8-sig", mode="w")
    handle.write(content)
    handle.close()
    return handle.name


# 快捷标签来自赛题手册第4页的业务场景及内容重点，仅辅助任务输入。
QUICK_TAGS = {
    "audience": ["大学新生", "家长", "商务人士", "移动办公者", "设计创作者", "摄影与视频创作者", "家庭用户", "中小企业采购", "附近潜在顾客", "新品关注者"],
    "scene": ["开学季", "职场办公", "差旅与移动办公", "内容创作", "家庭换新", "企业采购", "门店到访", "新品首发", "节日送礼"],
    "requirements": ["突出学习、携带与续航", "强调多设备协同", "聚焦会议与文档处理", "围绕屏幕与创作流程", "强调易用与数据迁移", "一个核心卖点＋到店引导", "不写无依据价格与补贴", "不伪造个人体验", "不虚构软件测试结果"],
}
CUSTOM_TAG = "＋ 自定义"

# ═══ 大白话录入：规则解析（不依赖 LLM，离线可用）+ 选择题式追问 ═══
PLATFORM_HINTS = [("小红书", "小红书"), ("公众号", "公众号"), ("朋友圈", "门店朋友圈"), ("海报", "门店海报")]
AUDIENCE_HINTS = [("新生", "大学新生"), ("学生", "大学新生"), ("家长", "家长"), ("商务", "商务人士"),
                  ("上班族", "商务人士"), ("职场", "商务人士"), ("差旅", "移动办公者"), ("出差", "移动办公者"),
                  ("设计师", "设计创作者"), ("摄影", "摄影与视频创作者"), ("剪辑", "摄影与视频创作者"),
                  ("创作", "设计创作者"), ("家庭", "家庭用户"), ("企业", "中小企业采购"), ("采购", "中小企业采购")]
SCENE_HINTS = [("开学", "开学季"), ("职场", "职场办公"), ("办公", "职场办公"), ("出差", "差旅与移动办公"),
               ("差旅", "差旅与移动办公"), ("剪辑", "内容创作"), ("创作", "内容创作"), ("换新", "家庭换新"),
               ("到店", "门店到访"), ("新品", "新品首发"), ("送礼", "节日送礼")]
FIELD_LABELS = {"audience": "目标人群", "scene": "使用场景", "platform": "发布平台"}


def _free_parse(text: str) -> Dict[str, str]:
    """从大白话里抽取 产品/人群/场景/平台；型号按最长匹配防 Pro 与 Pro S 串线。"""
    text = _safe_text(text)
    fills: Dict[str, str] = {}
    normalize = lambda value: re.sub(r"\s+", "", value).lower()
    compact = normalize(text)
    best = ""
    for model in load_model_choices():
        short = normalize(re.sub(r"^HUAWEI\s*", "", model, flags=re.I))
        key = normalize(model)
        best_short = normalize(re.sub(r"^HUAWEI\s*", "", best, flags=re.I))
        if (key in compact or short in compact) and len(short) > len(best_short):
            fills["product"] = model
            best = model
    for keyword, value in AUDIENCE_HINTS:
        if keyword in text and "audience" not in fills:
            fills["audience"] = value
            break
    for keyword, value in SCENE_HINTS:
        if keyword in text and "scene" not in fills:
            fills["scene"] = value
            break
    for keyword, value in PLATFORM_HINTS:
        if keyword in text:
            fills["platform"] = value
            break
    return fills


def _free_apply(state: Dict[str, Any]):
    """按已识别字段填表；缺失字段出选择题（一次一个，最多问完三项）。"""
    state = dict(state or {})
    missing = [key for key in ("audience", "scene", "platform") if not state.get(key)]
    updates = [
        gr.update(value=state["product"]) if state.get("product") else gr.skip(),
        gr.update(value=state.get("audience")) if state.get("audience") else gr.skip(),
        gr.update(value=state.get("scene")) if state.get("scene") else gr.skip(),
        gr.update(value=state.get("platform")) if state.get("platform") else gr.skip(),
    ]
    recognized = "、".join(f"{FIELD_LABELS.get(k, k)}={state[k]}" for k in ("product", "audience", "scene", "platform") if state.get(k))
    if missing:
        kind = missing[0]
        if kind == "platform":
            choices = ["小红书", "公众号", "门店朋友圈", "门店海报"]
        else:
            choices = _quick_tag_choices(kind, {})
        radio = gr.update(visible=True, value=None, choices=choices + ["跳过，保持现状"],
                          label=f"🤔 还差「{FIELD_LABELS[kind]}」，选一个：")
        feedback = f"✅ 已识别：{recognized or '（暂无）'}\n\n还差 **{FIELD_LABELS[kind]}**，点下方选择题补齐（可跳过，最多再问 {len(missing) - 1} 轮）。"
    else:
        radio = gr.update(visible=False, value=None, choices=[])
        feedback = f"✅ 需求已齐：{recognized}\n\n直接点下方「生成营销内容（3版并列）」。"
    return (*updates, feedback, radio, state)


def _free_start(text: str, state: Dict[str, Any]):
    if not _safe_text(text):
        return (gr.skip(),) * 4 + ("请先输入一句话，如：帮我推 MateBook Pro 给大学新生发小红书，开学季", gr.update(visible=False), state)
    state = dict(state or {})
    state.update(_free_parse(text))
    return _free_apply(state)


def _free_choose(choice: str, state: Dict[str, Any]):
    state = dict(state or {})
    choice = _safe_text(choice)
    if choice and choice != "跳过，保持现状":
        missing = [key for key in ("audience", "scene", "platform") if not state.get(key)]
        if missing:
            state[missing[0]] = choice
    return _free_apply(state)


def _quick_tag_choices(kind: str, usage: Dict[str, Any] | None) -> List[str]:
    """按浏览器累计点击次数降序排列；同频保留手册顺序。"""
    counts = (usage or {}).get(kind, {})
    labels = list(dict.fromkeys(QUICK_TAGS[kind] + list(counts)))
    labels.sort(key=lambda label: -counts.get(label, 0))
    return labels + ([CUSTOM_TAG] if kind in ("audience", "scene") else [])


def _use_quick_tag(kind: str, label: str, current: str, usage: Dict[str, Any] | None):
    """标签追加、不覆盖手工输入；重复点击只计数，不重复插入。"""
    updated = {key: dict(value) for key, value in (usage or {}).items()}
    counts = updated.setdefault(kind, {})
    counts[label] = counts.get(label, 0) + 1
    current = (current or "").strip()
    parts = [part.strip() for part in re.split(r"[；;\n]", current) if part.strip()]
    if label not in parts:
        current = f"{current}；{label}" if current else label
    return current, updated


def _quick_input(label: str, placeholder: str, kind: str, usage: gr.BrowserState, lines: int = 1):
    """输入框获焦才展开标签，组内点击不会触发提前收起。"""
    with gr.Group(elem_classes=["quick-field"]):
        field = gr.Textbox(label=label, placeholder=placeholder, lines=lines)
        with gr.Group(elem_classes=["quick-panel"]):
            tags = gr.Radio(_quick_tag_choices(kind, {}), value=None, show_label=False,
                            container=False, interactive=True, elem_classes=["quick-tags"])
            gr.Markdown("点击追加 · 常用标签优先", elem_classes=["quick-hint"])
            if kind in ("audience", "scene"):
                with gr.Group(visible=False) as custom_group:
                    custom = gr.Textbox(label=f"自定义{label}标签", placeholder="输入标签名称（最多60字）", max_lines=1)
                    save = gr.Button("保存并使用", size="sm")

    def choose(value, current, history):
        if not value:
            return (gr.skip(),) * (4 if kind != "requirements" else 3)
        if value == CUSTOM_TAG:
            return gr.skip(), gr.update(value=None), gr.skip(), gr.update(visible=True)
        text, history = _use_quick_tag(kind, value, current, history)
        result = (text, gr.update(choices=_quick_tag_choices(kind, history), value=None), history)
        return result + (gr.update(visible=False),) if kind != "requirements" else result

    targets = [field, tags, usage]
    if kind != "requirements":
        targets.append(custom_group)

        def save_custom(value, current, history):
            value = (value or "").strip()
            if not value or len(value) > 60 or value == CUSTOM_TAG or re.search(r"[；;\r\n]", value):
                gr.Warning("请输入1–60字的单个标签，不含分号或换行。")
                return (gr.skip(),) * 5
            text, history = _use_quick_tag(kind, value, current, history)
            return text, gr.update(choices=_quick_tag_choices(kind, history), value=None), history, gr.update(visible=False), ""

        save.click(save_custom, [custom, field, usage], targets + [custom], queue=False)
        custom.submit(save_custom, [custom, field, usage], targets + [custom], queue=False)
    tags.input(choose, [tags, field, usage], targets, queue=False)
    return field, tags


def _render_scene_ui(scene_text: str) -> str:
    """把既有场景判断文本排成四个轻量标签，不改变其中的业务内容。"""
    escape = html_lib.escape
    values: Dict[str, str] = {}
    for line in scene_text.splitlines():
        match = re.match(r"-\s*([^：:]+)[：:]\s*(.*)", line.strip())
        if match:
            values[match.group(1).strip()] = match.group(2).strip()
    chip_keys = ("目标人群", "使用场景", "内容目标", "平台")
    chips = "".join(
        f'<div class="scene-chip"><span>{escape(key)}</span><strong>{escape(values.get(key, "待确认"))}</strong></div>'
        for key in chip_keys
    )
    question = values.get("用户问题", "")
    note = f'<p class="scene-note"><span>判断提示</span>{escape(question)}</p>' if question else ""
    return f'<div class="scene-chip-grid">{chips}</div>{note}'


def _render_verification_ui(verification_text: str) -> str:
    """把既有核验文本做状态化展示；判断仅用于样式和计数。"""
    escape = html_lib.escape
    main_items: List[Tuple[str, str]] = []
    trace_items: List[Tuple[str, str]] = []
    current_trace = -1
    in_trace = False
    for raw_line in verification_text.splitlines():
        line = raw_line.strip()
        if line.startswith("#### 产品事实逐条回查"):
            in_trace = True
            continue
        if in_trace and line.startswith("- ✅ "):
            trace_items.append((line[4:].strip(), ""))
            current_trace = len(trace_items) - 1
        elif in_trace and line.startswith(">") and current_trace >= 0:
            text, _ = trace_items[current_trace]
            trace_items[current_trace] = (text, line.lstrip("> ").strip())
        elif line.startswith("- "):
            item = line[2:].strip()
            if "：" in item:
                label, value = item.split("：", 1)
                main_items.append((label.strip(), value.strip()))

    def is_pending(label: str, value: str) -> bool:
        return label == "待确认信息" or any(word in value for word in ("待人工确认", "需人工核对", "未找到", "禁止使用"))

    passed = sum(1 for label, value in main_items if label != "资料模式" and not is_pending(label, value))
    pending = sum(1 for label, value in main_items if is_pending(label, value))
    rows = []
    animation_index = 0
    for label, value in main_items:
        if label == "资料模式":
            rows.append(
                f'<div class="verify-row neutral"><span class="verify-dot"></span><div><strong>{escape(label)}</strong><p>{escape(value)}</p></div></div>'
            )
            continue
        warning = is_pending(label, value)
        if warning:
            icon = '<span class="warning-icon" aria-hidden="true">!</span>'
            row_class = "verify-row warning"
            delay = ""
        else:
            icon = ('<span class="check-icon" aria-hidden="true"><svg viewBox="0 0 24 24">'
                    '<path d="M5 12.5l4.2 4.2L19 7"/></svg></span>')
            row_class = "verify-row passed"
            delay = f' style="--check-delay:{animation_index * 0.15:.2f}s"'
            animation_index += 1
        rows.append(
            f'<div class="{row_class}"{delay}>{icon}<div><strong>{escape(label)}</strong><p>{escape(value)}</p></div></div>'
        )
    traces = "".join(
        '<div class="trace-row"><span class="mini-check">✓</span><div>'
        f'<strong>{escape(text)}</strong>'
        + (f'<p>{escape(source)}</p>' if source else "") + '</div></div>'
        for text, source in trace_items
    )
    if not traces:
        traces = '<p class="empty-note">未取得可回查的本方事实，不输出数字参数。</p>'
    return (
        f'<div class="verification-summary"><strong>{passed} 项通过</strong><span>·</span><strong>{pending} 项待确认</strong></div>'
        f'<div class="verification-list">{"".join(rows)}</div>'
        f'<div class="trace-group"><h4>产品事实逐条回查</h4>{traces}</div>'
    )


def build_demo() -> gr.Blocks:
    model_choices = list(load_model_choices())
    css = """
    :root {
        color-scheme: dark;
        --space-1:#0B1437; --space-2:#16224A; --space-3:#1B2A4A;
        --champagne:#C9A96A; --warm:#F5F0E6; --ink:#1F2329;
        --muted:#667085; --red:#C7000B; --red-light:#E63946;
        --green:#3EB575; --amber:#D48806;
    }
    html, body {
        min-height:100%; background:linear-gradient(160deg,#0B1437 0%,#16224A 55%,#1B2A4A 100%) fixed !important;
        font-family:"PingFang SC","Microsoft YaHei",sans-serif !important;
    }
    body { margin:0; }
    .gradio-container, .gradio-container.dark {
        max-width:none !important; min-height:100vh; margin:0 !important; padding:0 !important;
        background:transparent !important; color:var(--warm);
    }
    .app-shell { width:min(1080px,calc(100% - 40px)); margin:0 auto; padding:52px 0 64px; position:relative; z-index:2; gap:24px !important; }
    .ambient-bubbles { position:fixed; inset:0; overflow:hidden; pointer-events:none; z-index:0; }
    .ambient-bubbles span { position:absolute; border-radius:50%; filter:blur(60px); opacity:.35; animation:float 10s ease-in-out infinite alternate; }
    .ambient-bubbles span:nth-child(1){width:340px;height:340px;left:-90px;top:5%;background:radial-gradient(circle,#4A6CF7,#7C5CFF);animation-duration:10s}
    .ambient-bubbles span:nth-child(2){width:260px;height:260px;right:4%;top:22%;background:radial-gradient(circle,#2BB5A0,#4A6CF7);animation-duration:14s;animation-delay:-4s}
    .ambient-bubbles span:nth-child(3){width:300px;height:300px;left:18%;bottom:-80px;background:radial-gradient(circle,#7C5CFF,#4A6CF7);animation-duration:12s;animation-delay:-7s}
    .ambient-bubbles span:nth-child(4){width:210px;height:210px;right:22%;bottom:12%;background:radial-gradient(circle,#2BB5A0,#245EDB);animation-duration:8s;animation-delay:-2s}
    @keyframes float { from{transform:translateY(-20px) scale(.98)} to{transform:translateY(20px) scale(1.04)} }

    .hero { padding:10px 4px 20px; }
    .hero-eyebrow { color:var(--champagne); font-size:13px; letter-spacing:0; margin:0 0 10px; }
    .hero h1 { color:var(--warm); font-size:54px; line-height:1.12; letter-spacing:0; margin:0; font-weight:750; }
    .hero-rule { width:92px; height:1px; margin:20px 0 16px; background:var(--champagne); box-shadow:0 0 12px rgba(201,169,106,.45); }
    .hero p { color:rgba(245,240,230,.78); max-width:720px; font-size:16px; line-height:1.7; margin:0; }
    .hero .source-promise { margin-top:8px; color:rgba(245,240,230,.56); font-size:13px; }

    .control-panel, .modify-panel, .prompt-panel, .retrieval-panel {
        background:rgba(255,255,255,.08) !important; backdrop-filter:blur(18px) saturate(135%);
        -webkit-backdrop-filter:blur(18px) saturate(135%); border:1px solid rgba(201,169,106,.42) !important;
        border-radius:16px !important; box-shadow:0 8px 32px rgba(0,0,0,.22) !important;
    }
    .control-panel { padding:24px !important; overflow:visible !important; }
    .control-panel > .styler, .modify-panel > .styler, .result-card > .styler {
        background:transparent !important; border:0 !important; border-radius:inherit !important;
        box-shadow:none !important; overflow:visible !important;
    }
    .control-panel .block, .control-panel .form, .control-panel .wrap,
    .modify-panel .block, .modify-panel .form, .modify-panel .wrap {
        background:transparent !important; border-color:transparent !important; box-shadow:none !important;
    }
    .panel-heading { margin-bottom:16px; }
    .panel-heading span { display:block; color:var(--champagne); font-size:12px; margin-bottom:4px; }
    .panel-heading h2 { color:var(--warm); font-size:20px; margin:0; letter-spacing:0; }
    .input-grid { gap:18px !important; overflow:visible !important; }
    .input-column { gap:12px !important; overflow:visible !important; }
    .input-column > div, .quick-field, .quick-panel, .quick-tags { overflow:visible !important; }
    .control-panel label, .modify-panel label { color:rgba(245,240,230,.9) !important; font-weight:600; }
    .control-panel input, .control-panel textarea, .modify-panel input, .modify-panel textarea {
        background:rgba(8,17,48,.72) !important; color:var(--warm) !important;
        border:1px solid rgba(201,169,106,.35) !important; border-radius:10px !important;
    }
    .control-panel input::placeholder, .control-panel textarea::placeholder,
    .modify-panel input::placeholder, .modify-panel textarea::placeholder { color:rgba(245,240,230,.42) !important; }
    .control-panel input:focus, .control-panel textarea:focus, .modify-panel input:focus, .modify-panel textarea:focus {
        border-color:var(--champagne) !important; box-shadow:0 0 0 2px rgba(201,169,106,.16) !important;
    }
    /* 仅做颜色与边框限定，不修改 Dropdown 的 appearance、pointer-events 或 overflow。 */
    .platform-select, .product-select, .revision-select { position:relative; z-index:6; }
    .platform-select input, .product-select input, .revision-select input { cursor:pointer; }
    .platform-select:focus-within, .revision-select:focus-within { z-index:10000; }
    [role="listbox"] { z-index:9999 !important; background:#111c3d !important; border:1px solid rgba(201,169,106,.55) !important; color:var(--warm) !important; }
    [role="option"] { color:var(--warm) !important; background:#111c3d !important; }
    [role="option"]:hover, [role="option"][aria-selected="true"] { background:rgba(201,169,106,.22) !important; color:#fff8e9 !important; }

    button.generate-btn, button.export-btn, .generate-btn > button, .export-btn > button {
        min-height:48px; background:linear-gradient(135deg,var(--red),var(--red-light)) !important;
        color:white !important; border:0 !important; border-radius:12px !important; font-weight:700 !important;
        box-shadow:0 8px 22px rgba(199,0,11,.32) !important; transition:transform .2s ease,box-shadow .2s ease !important;
    }
    button.generate-btn:hover, button.export-btn:hover, .generate-btn > button:hover, .export-btn > button:hover { transform:translateY(-2px); box-shadow:0 12px 28px rgba(199,0,11,.42) !important; }
    button.generate-btn:disabled, .generate-btn > button:disabled { opacity:.72; cursor:wait; transform:none; }
    button.generate-btn:disabled::before, .generate-btn > button:disabled::before { content:""; width:15px; height:15px; border:2px solid rgba(255,255,255,.4); border-top-color:#fff; border-radius:50%; animation:spin .8s linear infinite; }
    @keyframes spin { to{transform:rotate(360deg)} }
    button.secondary-btn, .secondary-btn > button { background:rgba(255,255,255,.08) !important; color:var(--warm) !important; border:1px solid rgba(201,169,106,.5) !important; border-radius:12px !important; }

    .status-box { padding:12px 16px !important; background:rgba(4,12,39,.52) !important; border:1px solid rgba(201,169,106,.32) !important; border-radius:12px !important; }
    .status-box p { margin:0 !important; color:rgba(245,240,230,.82) !important; font-size:13px; }
    .status-box strong { color:var(--warm) !important; }
    .loading-steps { display:flex; gap:8px; flex-wrap:wrap; color:rgba(245,240,230,.7); font-size:13px; padding:6px 0; }
    .loading-steps span { animation:stepPulse 2.4s ease-in-out infinite; }
    .loading-steps span:nth-child(2){animation-delay:.5s}.loading-steps span:nth-child(3){animation-delay:1s}.loading-steps span:nth-child(4){animation-delay:1.5s}
    @keyframes stepPulse { 0%,28%,100%{color:rgba(245,240,230,.42)} 12%{color:var(--champagne);text-shadow:0 0 10px rgba(201,169,106,.65)} }

    .quick-field { background:transparent !important; border:0 !important; box-shadow:none !important; gap:0 !important; }
    .quick-field .quick-panel { display:none !important; background:transparent !important; border:0 !important; box-shadow:none !important; padding:10px 0 4px !important; height:auto !important; max-height:none !important; overflow:visible !important; }
    .quick-field:focus-within .quick-panel, .quick-field .quick-panel:hover { display:flex !important; flex-direction:column; gap:8px !important; }
    .quick-tags fieldset, .quick-tags .wrap { display:flex !important; flex-wrap:wrap !important; gap:10px !important; height:auto !important; max-height:none !important; overflow:visible !important; padding:0 !important; border:0 !important; background:transparent !important; }
    .quick-tags label { width:auto !important; flex:0 0 auto !important; border-radius:999px !important; padding:7px 12px !important; background:rgba(255,255,255,.07) !important; border:1px solid rgba(245,240,230,.32) !important; color:rgba(245,240,230,.9) !important; cursor:pointer; transition:.18s ease; }
    .quick-tags label:hover, .quick-tags label:has(input:checked) { border-color:var(--champagne) !important; color:#FFE3A7 !important; box-shadow:0 0 12px rgba(201,169,106,.22); background:rgba(201,169,106,.10) !important; }
    .quick-tags input[type="radio"] { position:absolute; opacity:0; width:1px; height:1px; }
    .quick-tags label:focus-within { outline:2px solid var(--champagne); outline-offset:2px; }
    .quick-hint { background:transparent !important; border:0 !important; padding:0 !important; }
    .quick-hint p { color:rgba(245,240,230,.5) !important; font-size:11px; margin:0 !important; }

    .retrieval-panel, .prompt-panel { overflow:visible !important; }
    .retrieval-panel summary, .prompt-panel summary { color:var(--warm) !important; padding:14px 18px !important; }
    .retrieval-panel > div, .prompt-panel > div { color:rgba(245,240,230,.84); }
    .prompt-panel pre, .prompt-panel code { background:#09122e !important; color:#e8e2d7 !important; }
    .prompt-panel small { color:rgba(245,240,230,.52); }

    .results-heading { margin:18px 0 0; }
    .results-heading span { color:var(--champagne); font-size:12px; }
    .results-heading h2 { color:var(--warm); font-size:26px; margin:4px 0 0; }
    .results-stack { gap:24px !important; }
    .result-card {
        position:relative; padding:28px !important; background:rgba(255,255,255,.92) !important;
        backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
        border:1px solid rgba(201,169,106,.45) !important; border-radius:16px !important;
        box-shadow:0 8px 32px rgba(0,0,0,.25) !important; color:var(--ink) !important; overflow:hidden;
    }
    .result-card::before { content:""; position:absolute; left:0; right:0; top:0; height:2px; background:linear-gradient(90deg,transparent,var(--champagne),transparent); }
    .card-heading { display:flex; align-items:center; gap:12px; padding-bottom:15px; margin-bottom:18px; border-bottom:1px solid rgba(201,169,106,.28); }
    .card-heading::before { content:""; width:3px; height:26px; background:var(--champagne); border-radius:2px; }
    .card-heading span { color:#8b6b31; font-family:"DIN Alternate","Helvetica Neue",sans-serif; font-size:12px; font-weight:700; }
    .card-heading h3 { color:var(--ink); font-size:18px; margin:0; letter-spacing:0; }

    .scene-card { padding:20px 24px !important; }
    .scene-card .card-heading { display:none; }
    .scene-chip-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
    .scene-chip { min-width:0; padding:10px 13px; border:1px solid rgba(201,169,106,.62); border-radius:999px; background:rgba(201,169,106,.07); }
    .scene-chip span { display:block; color:#85714e; font-size:11px; margin-bottom:2px; }
    .scene-chip strong { display:block; color:var(--ink); font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .scene-note { color:#687080; font-size:12px; line-height:1.65; margin:12px 4px 0; }
    .scene-note span { color:#8b6b31; font-weight:700; margin-right:8px; }

    .basis-heading { display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin-bottom:14px; color:#657080; font-size:12px; }
    .basis-heading strong { color:var(--ink); font-size:15px; }
    .fact-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:24px; }
    .fact-stat { min-width:0; padding:16px; border:1px solid rgba(201,169,106,.36); border-radius:12px; background:linear-gradient(145deg,rgba(255,255,255,.8),rgba(249,246,238,.72)); }
    .fact-label { display:block; color:#777e89; font-size:12px; margin-bottom:7px; }
    .fact-value { display:block; color:#A78342; font-family:"DIN Alternate","Helvetica Neue",sans-serif; font-size:26px; line-height:1.1; letter-spacing:0; overflow-wrap:anywhere; }
    .talk-group, .evidence-group, .image-group, .trace-group { margin-top:22px; padding-top:18px; border-top:1px solid rgba(201,169,106,.26); }
    .talk-group h4, .evidence-group h4, .image-group h4, .trace-group h4 { color:var(--ink); font-size:15px; margin:0 0 12px; }
    .talk-quote { position:relative; margin:0 0 12px; padding:14px 17px 14px 20px; border-left:3px solid var(--champagne); background:rgba(201,169,106,.08); color:#343941; font-style:italic; }
    .talk-kicker { color:#8b6b31; font-size:11px; font-style:normal; font-weight:700; margin-bottom:6px; }
    .talk-quote p { margin:4px 0; line-height:1.7; font-size:13px; }
    .talk-quote footer { color:#747b86; font-size:11px; font-style:normal; margin-top:7px; }
    .evidence-list { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
    .evidence-card { position:relative; padding:15px; background:rgba(247,247,246,.82); border:1px solid #e4ded1; border-radius:10px; }
    .evidence-title { color:var(--ink); font-size:13px; line-height:1.5; }
    .evidence-source { color:#7b818b; font-size:11px; margin:4px 0 9px; }
    .evidence-excerpt { color:#505761; font-size:12px; line-height:1.75; }
    .evidence-excerpt > span { display:block; color:#8b6b31; font-size:10px; font-weight:700; margin-bottom:4px; }
    .evidence-mark { color:inherit; background:linear-gradient(transparent 60%,#FFE58F 60%,#FFE58F 90%,transparent 90%); }
    .evidence-card a, .talk-quote a, .image-group a { color:#8A621D !important; font-size:11px; text-decoration:none; }
    .competitor-card { opacity:.6; border-color:#aab0ba; filter:grayscale(.25); }
    .reference-badge { position:absolute; right:10px; top:10px; padding:3px 7px; border-radius:999px; background:#e3e5e8; color:#555d68; font-size:9px; }
    .image-help { color:#7b818b; font-size:11px; margin:-7px 0 12px; }
    .image-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
    .image-grid figure { margin:0; min-width:0; border-radius:12px; overflow:hidden; background:#f2f2f0; border:1px solid #e2ddd2; }
    .image-grid img { display:block; width:100%; max-width:320px; aspect-ratio:4/3; object-fit:contain; background:white; }
    .image-grid figure > p { min-height:120px; display:grid; place-items:center; padding:12px; color:#7b818b; font-size:11px; }
    .image-grid figcaption { padding:10px 11px; color:#6d747f; font-size:10px; line-height:1.5; }
    .image-grid figcaption strong { color:var(--ink); font-size:11px; }
    .empty-note { color:#7b818b; font-size:12px; }

    .article-card { padding:38px 42px !important; }
    .article-markdown, .article-markdown .prose { color:var(--ink) !important; }
    .article-markdown h3 { color:#755928 !important; font-size:14px !important; margin:24px 0 8px !important; letter-spacing:0 !important; }
    .article-markdown h3:first-child { display:flex; align-items:center; gap:10px; margin-top:0 !important; }
    .article-markdown h3:first-child::before { content:""; width:3px; height:26px; background:var(--champagne); border-radius:2px; }
    .article-markdown h3:first-child + p { color:var(--ink) !important; font-size:22px !important; font-weight:750; line-height:1.45 !important; margin:8px 0 20px !important; }
    .article-markdown p, .article-markdown li { color:#30353c !important; font-size:15px; line-height:1.8 !important; }
    .article-markdown strong { color:#20242a !important; }
    .article-markdown h3:last-of-type ~ p { display:inline-block; border:1px solid rgba(201,169,106,.55); border-radius:999px; padding:5px 11px; background:rgba(201,169,106,.08); }

    .verification-summary { display:flex; align-items:center; gap:8px; color:#69717c; margin-bottom:15px; }
    .verification-summary strong:first-child { color:#27885a; }
    .verification-summary strong:last-child { color:var(--amber); }
    .verification-list { display:grid; gap:10px; }
    .verify-row { display:flex; align-items:flex-start; gap:11px; padding:11px 12px; border-radius:10px; background:#f7f7f5; }
    .verify-row strong { color:var(--ink); font-size:13px; }
    .verify-row p { color:#606873; font-size:12px; line-height:1.6; margin:2px 0 0; }
    .check-icon { width:22px; height:22px; flex:0 0 22px; border-radius:50%; background:rgba(62,181,117,.12); transform:scale(0); animation:checkPop .3s ease-out var(--check-delay) forwards; }
    .check-icon svg { width:22px; height:22px; fill:none; stroke:var(--green); stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; }
    .check-icon path { stroke-dasharray:24; stroke-dashoffset:24; animation:checkDraw .36s ease-out calc(var(--check-delay) + .08s) forwards; }
    @keyframes checkPop { 70%{transform:scale(1.12)} 100%{transform:scale(1)} }
    @keyframes checkDraw { to{stroke-dashoffset:0} }
    .warning { background:rgba(212,136,6,.08); }
    .warning-icon { display:grid; place-items:center; width:22px; height:22px; flex:0 0 22px; border-radius:50%; background:rgba(212,136,6,.16); color:var(--amber); font-weight:800; }
    .verify-dot { width:7px; height:7px; flex:0 0 7px; margin:7px; border-radius:50%; background:#98a0aa; }
    .trace-row { display:flex; gap:9px; padding:9px 0; border-bottom:1px solid #ebe7de; }
    .mini-check { color:var(--green); font-weight:800; }
    .trace-row strong { color:#343941; font-size:12px; }
    .trace-row p { color:#747b86; font-size:11px; line-height:1.55; margin:3px 0 0; }

    .modify-panel { padding:22px !important; }
    .modify-panel .prose h3 { color:var(--warm) !important; margin:0 0 12px; }
    .combined-output textarea { min-height:190px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace !important; font-size:11px !important; }
    .footer-note { text-align:center; margin-top:14px; }
    .footer-note p { color:rgba(245,240,230,.48) !important; font-size:11px; }

    .version-tabs .tab-nav button { font-weight:600 !important; }
    .publish-box textarea { font-size:13px !important; line-height:1.7 !important; }
    .fav-btn > button, button.fav-btn { min-height:32px !important; font-size:12px !important; }
    .center-panel { padding:22px !important; background:rgba(255,255,255,.08) !important; backdrop-filter:blur(18px); border:1px solid rgba(201,169,106,.42) !important; border-radius:16px !important; }
    .center-panel h4 { color:var(--champagne) !important; margin:18px 0 8px !important; }
    .center-panel label { color:rgba(245,240,230,.9) !important; }
    .record-item { padding:10px 12px; border:1px solid rgba(201,169,106,.3); border-radius:10px; margin-bottom:8px; background:rgba(4,12,39,.4); color:rgba(245,240,230,.85); font-size:12px; }
    .record-item summary { cursor:pointer; color:rgba(245,240,230,.85); }
    .record-preview { color:rgba(245,240,230,.5); margin-left:8px; }
    .record-body { white-space:pre-wrap; margin-top:8px; color:rgba(245,240,230,.75); line-height:1.7; max-height:220px; overflow:auto; }
    .fav-meta { color:var(--champagne); margin-bottom:4px; }
    .center-status p { color:var(--green) !important; font-size:13px; }

    @media (max-width:767px) {
        .app-shell { width:min(100% - 24px,1080px); padding-top:28px; }
        .ambient-bubbles span:nth-child(3), .ambient-bubbles span:nth-child(4) { display:none; }
        .hero h1 { font-size:34px; }
        .control-panel, .result-card, .article-card { padding:20px !important; }
        .scene-chip-grid, .fact-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .evidence-list, .image-grid { grid-template-columns:1fr; }
        .scene-chip strong { white-space:normal; }
    }
    """
    with gr.Blocks(title="AI+华为PC营销软文智能体", theme=gr.themes.Base(primary_hue="slate"), css=css) as demo:
        gr.HTML('<div class="ambient-bubbles" aria-hidden="true"><span></span><span></span><span></span><span></span></div>')
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML('''<header class="hero"><p class="hero-eyebrow">HUAWEI PC · KNOWLEDGE-DRIVEN CONTENT AGENT</p>
                <h1>AI+华为PC营销软文智能体</h1><div class="hero-rule"></div>
                <p>从场景解析到资料检索、图文编排、事实核验与导出的智能内容工作台。</p>
                <p class="source-promise">事实与图片均溯源自 ima 知识库</p></header>''')
            status = gr.Markdown("**知识库状态：等待检索**", elem_classes=["status-box"])
            # 仅存标签偏好，不含接口凭证；固定存储标识便于刷新/重启后复用。
            tag_usage = gr.BrowserState({}, storage_key="huawei-pc-quick-tags-v1", secret="huawei-pc-tag-preferences-v1")
            with gr.Group(elem_classes=["control-panel"]):
                gr.HTML('<div class="panel-heading"><span>01 / TASK BRIEF</span><h2>任务输入</h2></div>')
                with gr.Group():
                    free_input = gr.Textbox(
                        label="💡 一句话录入（推荐，不用填表）",
                        placeholder="跟我说说这次要推什么？如：帮我推 MateBook Pro 给大学新生发小红书，开学季",
                        lines=1)
                    free_parse_btn = gr.Button("理解并填好需求", elem_classes=["secondary-btn"])
                    free_feedback = gr.Markdown("")
                    free_choice = gr.Radio([], label="", visible=False)
                    free_state = gr.State({})
                with gr.Accordion("📋 完整表单（高级模式，已识别的会自动填好）", open=False):
                    with gr.Row(elem_classes=["input-grid"]):
                        with gr.Column(scale=1, elem_classes=["input-column"]):
                            product = gr.Dropdown(model_choices, value=model_choices[0], label="产品型号", allow_custom_value=True, elem_classes=["product-select"])
                            audience, audience_tags = _quick_input("目标人群", "如：高校新生、差旅销售、内容创作者", "audience", tag_usage)
                            scene, scene_tags = _quick_input("使用场景", "如：开学季、移动办公、视频剪辑", "scene", tag_usage)
                        with gr.Column(scale=1, elem_classes=["input-column"]):
                            # 平台固定为选项下拉，关闭输入搜索与自定义值。
                            platform = gr.Dropdown(["公众号", "小红书", "门店朋友圈", "门店海报"], value="公众号", label="发布平台", filterable=False, allow_custom_value=False, elem_classes=["platform-select"])
                            requirements, requirements_tags = _quick_input("补充要求", "如：强调便携，不写价格，语气克制", "requirements", tag_usage, lines=3)
                generate = gr.Button("生成营销内容（3版并列）", variant="primary", elem_classes=["generate-btn"])
                with gr.Accordion("高级选项", open=False):
                    attachment = gr.File(label="📎 上传参考附件（.md/.txt，仅供风格参考，事实仍以 ima 为准）",
                                         file_types=[".md", ".txt"], type="filepath")

            with gr.Accordion("检索过程与命中资料", open=False, elem_classes=["retrieval-panel"]):
                retrieval = gr.HTML("点击生成后显示命中的资料标题、关键摘录和图片。")

            # 直接展示实际用于模型调用的常量，不另存副本，也不参与生成事件。
            with gr.Accordion("🧠 查看本 Agent 的自研系统提示词（7大业务模块，赛题评分项）", open=False, elem_classes=["prompt-panel"]):
                gr.Markdown("本提示词内置于产品，随每次生成任务实时注入大模型调用，覆盖：角色与任务 / 场景理解 / 平台规则 / 知识库约束 / 图片规则 / 生成要求 / 核验+合规 七大模块。")
                gr.Code(value=SYSTEM_PROMPT, language="markdown", interactive=False,
                        label="实际注入模型的 SYSTEM_PROMPT", wrap_lines=True, lines=12)
                gr.HTML("<small>完整文档见 GitHub 仓库 docs/系统提示词.md</small>")

            gr.HTML('<div class="results-heading"><span>02 / VERIFIED DELIVERABLE</span><h2>四段式交付结果</h2></div>')
            with gr.Column(elem_classes=["results-stack"]):
                with gr.Group(elem_classes=["result-card", "scene-card"]):
                    gr.HTML('<div class="card-heading"><span>01</span><h3>场景判断</h3></div>')
                    scene_output = gr.HTML('<p class="empty-note">生成后展示人群、场景、目标与平台判断。</p>')
                with gr.Group(elem_classes=["result-card", "basis-card"]):
                    gr.HTML('<div class="card-heading"><span>02</span><h3>知识库依据</h3></div>')
                    basis_output = gr.HTML('<p class="empty-note">生成后展示事实点、话术依据、资料摘录与图片来源。</p>')
                with gr.Group(elem_classes=["result-card", "article-card"]):
                    gr.HTML('<div class="card-heading"><span>03</span><h3>图文软文（三版并列，挑顺眼的直接复制去发）</h3></div>')
                    versions_state = gr.State({"versions": [], "product": "", "platform": ""})
                    with gr.Tabs(elem_classes=["version-tabs"]):
                        variant_md, variant_pub, fav_btns, fav_msgs = [], [], [], []
                        for idx, vname in enumerate(VARIANT_NAMES):
                            with gr.Tab(f"版本{chr(65 + idx)} · {vname}"):
                                md = gr.Markdown("生成后展示该版本成稿。", elem_classes=["article-markdown"])
                                pub = gr.Textbox(label="📋 发布版（已净化 markdown，点右上角一键复制，粘贴即可发布）",
                                                 lines=6, show_copy_button=True, elem_classes=["publish-box"])
                                fav = gr.Button(f"⭐ 收藏「{vname}」此版", size="sm", elem_classes=["secondary-btn", "fav-btn"])
                                fav_msg = gr.Markdown("")
                                variant_md.append(md); variant_pub.append(pub); fav_btns.append(fav); fav_msgs.append(fav_msg)
                        gr.HTML('<small>小红书/朋友圈无公开发布接口，复制后请到 App 内粘贴发布。</small>')
                with gr.Group(elem_classes=["result-card", "verification-card"]):
                    gr.HTML('<div class="card-heading"><span>04</span><h3>核验结果</h3></div>')
                    verification_output = gr.HTML('<p class="empty-note">生成后逐项核验事实、图片与合规要求。</p>')

            with gr.Group(elem_classes=["modify-panel"]):
                gr.Markdown("### 修改与导出")
                with gr.Row():
                    revision_platform = gr.Dropdown(["公众号", "小红书", "门店朋友圈", "门店海报"], value="公众号", label="换平台重新生成", filterable=False, allow_custom_value=False, elem_classes=["revision-select"])
                    revision = gr.Textbox(label="临时修改要求", lines=2, placeholder="如：改成更适合门店朋友圈的短文案")
                regenerate = gr.Button("按修改要求重新生成", elem_classes=["secondary-btn"])
                combined = gr.Textbox(label="完整 Markdown（可复制）", lines=8, show_copy_button=True, elem_classes=["combined-output"])
                with gr.Row():
                    export = gr.Button("导出 Markdown 文件", variant="primary", elem_classes=["export-btn"])
                    export_txt = gr.Button("导出发布版 TXT（3版合集，记事本可读）", elem_classes=["secondary-btn"])
                download = gr.File(label="下载文件")

            with gr.Accordion("👤 个人中心（资料 / 历史 / 收藏 / 素材库）", open=False):
                with gr.Group(elem_classes=["center-panel"]):
                    gr.Markdown("#### 个人资料（用于自动落款与千店不撞稿）")
                    with gr.Row():
                        pf_name = gr.Textbox(label="姓名/昵称", placeholder="如：小林")
                        pf_store_type = gr.Dropdown(["华为智能生活馆", "华为授权体验店", "授权服务中心", "其他"], value="华为智能生活馆", label="门店类型")
                        pf_store = gr.Textbox(label="门店名称", placeholder="如：杭州湖滨银泰店")
                    with gr.Row():
                        pf_city = gr.Textbox(label="城市", placeholder="如：杭州")
                        pf_wecom = gr.Textbox(label="企业微信号（选填）")
                        pf_xhs = gr.Textbox(label="小红书账号（选填）")
                    pf_template = gr.Textbox(label="自定义落款模板（留空自动拼装）", placeholder="如：杭州湖滨银泰华为店 · 找小林报小红书有惊喜")
                    with gr.Row():
                        pf_auto_sign = gr.Checkbox(value=True, label="✍️ 生成时自动添加门店落款（朋友圈/海报/小红书）")
                        pf_memory = gr.Checkbox(value=True, label="🧠 生成时参考我的收藏风格（只影响语感，不作事实依据）")
                    pf_save = gr.Button("保存个人资料", elem_classes=["secondary-btn"])
                    pf_status = gr.Markdown("", elem_classes=["center-status"])

                    gr.Markdown("#### 📜 历史记录（每次生成自动存档）")
                    history_view = gr.HTML(_records_html(USER_DATA / "history.jsonl", "history"))
                    history_refresh = gr.Button("刷新历史", size="sm", elem_classes=["secondary-btn"])

                    gr.Markdown("#### ⭐ 收藏夹")
                    fav_view = gr.HTML(_records_html(USER_DATA / "favorites.jsonl", "fav"))
                    fav_refresh = gr.Button("刷新收藏", size="sm", elem_classes=["secondary-btn"])

                    gr.Markdown("#### 🖼️ 图片素材库（门店实拍，强制型号核验后入库）")
                    with gr.Row():
                        asset_model = gr.Dropdown(model_choices, value=model_choices[0], label="对应产品型号（必选，与 P1 型号矩阵同源核验）")
                        asset_note = gr.Textbox(label="拍摄说明（选填）", placeholder="如：展台实拍晴蓝配色")
                    asset_file = gr.File(label="上传门店实物照片（jpg/png/webp，≤10MB）", file_types=[".jpg", ".jpeg", ".png", ".webp"], type="filepath")
                    asset_upload = gr.Button("核验并入库", elem_classes=["secondary-btn"])
                    asset_status = gr.Markdown("", elem_classes=["center-status"])
                    asset_gallery = gr.HTML(gallery_html())
                    asset_refresh = gr.Button("刷新素材库", size="sm", elem_classes=["secondary-btn"])
            gr.Markdown("本作品由自研 Agent 驱动 · 事实与图片均检索自 ima 共享知识库 · 浙客松 2026", elem_classes=["footer-note"])

        outputs = [status, retrieval, scene_output, basis_output,
                   variant_md[0], variant_pub[0], variant_md[1], variant_pub[1], variant_md[2], variant_pub[2],
                   verification_output, combined, versions_state]

        def restore_tags(history):
            return tuple(gr.update(choices=_quick_tag_choices(kind, history), value=None) for kind in QUICK_TAGS)

        # 本地偏好恢复后以及每次使用标签后，同步三组排序。
        demo.load(restore_tags, [tag_usage], [audience_tags, scene_tags, requirements_tags], queue=False)
        tag_usage.change(restore_tags, [tag_usage], [audience_tags, scene_tags, requirements_tags], queue=False)

        # 大白话录入：解析填表 + 选择题式追问（不依赖 LLM，离线可用）。
        free_outputs = [product, audience, scene, platform, free_feedback, free_choice, free_state]
        free_parse_btn.click(_free_start, [free_input, free_state], free_outputs, queue=False)
        free_input.submit(_free_start, [free_input, free_state], free_outputs, queue=False)
        free_choice.input(_free_choose, [free_choice, free_state], free_outputs, queue=False)

        def adapt_result(status_text: str, basis_text: str, scene_text: str, basis_text2: str,
                         versions: List[str], verification_text: str, combined_text: str,
                         product_name: str = "", platform_name: str = ""):
            versions = (versions or []) + ["", ""]
            pubs = [_publish_text(v) if v and not v.startswith("⏳") else "" for v in versions[:3]]
            mds = [v if v else "（本版本未生成：离线降级模式或无可用凭证时仅输出单版本）" for v in versions[:3]]
            state = {"versions": versions[:3], "product": product_name, "platform": platform_name}
            return (status_text, basis_text, _render_scene_ui(scene_text), basis_text2,
                    mds[0], pubs[0], mds[1], pubs[1], mds[2], pubs[2],
                    _render_verification_ui(verification_text), combined_text, state)

        loading_html = ('<div class="loading-steps"><span>正在检索 ima 知识库</span><b>→</b>'
                        '<span>提取事实</span><b>→</b><span>生成软文</span><b>→</b><span>核验中</span></div>')

        def begin_generation():
            return gr.update(value="生成中…", interactive=False), _honor_loading_html("retrieve")

        def finish_generation():
            return gr.update(value="生成营销内容（3版并列）", interactive=True)

        def generate_with_retrieval(p, a, s, pl, req, attach):
            # 流式：依据区先亮、三版先完成先上屏；等待区用"王者加载"动画随阶段轮换文案。
            for snap in generate_flow_stream(p, a, s, pl, req, attachment_text=_read_attachment(attach)):
                adapted = list(adapt_result(*snap, product_name=p, platform_name=pl))
                if snap[5].startswith("### 四、核验结果\n三版本写作中"):
                    adapted[1] = _honor_loading_html("generate", done_count=0)
                elif "其余写作中" in snap[5]:
                    done_count = len([v for v in snap[4] if v and not v.startswith("⏳")])
                    adapted[1] = _honor_loading_html("generate", done_count=done_count)
                yield tuple(adapted)

        generate_event = generate.click(begin_generation, None, [generate, retrieval], queue=False)
        generate_event = generate_event.then(generate_with_retrieval, [product, audience, scene, platform, requirements, attachment], outputs)
        generate_event.then(finish_generation, None, [generate], queue=False)

        def regenerate_with_platform(p, a, s, pl, req, new_pl, rev, attach):
            result = list(generate_flow(p, a, s, new_pl or pl, req, rev, attachment_text=_read_attachment(attach)))
            target_platform = new_pl or pl
            if target_platform != pl and result[4] and result[4][0]:
                change_note = (
                    f"\n\n### 本版相对上版的变化点\n"
                    f"- 平台从「{pl}」切换为「{target_platform}」，正文结构、语气、行动引导和配图数量按新平台规则重排。"
                )
                result[4][0] += change_note
                result[6] = result[6].replace(
                    f"### 三、图文软文\n{result[4][0][:-len(change_note)]}",
                    f"### 三、图文软文\n{result[4][0]}",
                )
            return adapt_result(*result, product_name=p, platform_name=target_platform)

        regenerate.click(regenerate_with_platform, [product, audience, scene, platform, requirements, revision_platform, revision, attachment], outputs)
        export.click(export_markdown, [combined], [download])
        export_txt.click(export_publish_txt, [versions_state, platform], [download])

        for idx, fav in enumerate(fav_btns):
            fav.click(lambda state, i=idx: save_favorite(i, state), [versions_state], [fav_msgs[idx]], queue=False)

        def _save_profile(name, store_type, store, city, wecom, xhs, template, auto_sign, memory):
            return save_profile({"name": name, "store_type": store_type, "store_name": store, "city": city,
                                 "wecom": wecom, "xhs": xhs, "sign_template": template,
                                 "auto_signature": auto_sign, "memory_on": memory})

        pf_save.click(_save_profile,
                      [pf_name, pf_store_type, pf_store, pf_city, pf_wecom, pf_xhs, pf_template, pf_auto_sign, pf_memory],
                      [pf_status], queue=False)

        def _load_profile_ui():
            profile = load_profile()
            return (profile.get("name", ""), profile.get("store_type", "华为智能生活馆"), profile.get("store_name", ""),
                    profile.get("city", ""), profile.get("wecom", ""), profile.get("xhs", ""),
                    profile.get("sign_template", ""), profile.get("auto_signature", True), profile.get("memory_on", True))

        demo.load(_load_profile_ui, None,
                  [pf_name, pf_store_type, pf_store, pf_city, pf_wecom, pf_xhs, pf_template, pf_auto_sign, pf_memory], queue=False)
        history_refresh.click(lambda: _records_html(USER_DATA / "history.jsonl", "history"), None, [history_view], queue=False)
        fav_refresh.click(lambda: _records_html(USER_DATA / "favorites.jsonl", "fav"), None, [fav_view], queue=False)
        asset_upload.click(upload_store_image, [asset_model, asset_file, asset_note], [asset_status, asset_gallery], queue=False)
        asset_refresh.click(gallery_html, None, [asset_gallery], queue=False)
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
