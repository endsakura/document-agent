"""标题检测与文档类型推断（基于 OCR 结果 + 规则）。"""
import re
from typing import List, Optional, Tuple

from knowledge.rules_data import TITLE_RULES
from tools import ocr as ocr_module

NON_CONTRACT_AGREEMENT_TITLE_MARKERS = (
    "补充协议", "变更协议", "解除协议", "终止协议",
    "保密协议", "廉洁协议", "廉政协议",
    "承诺函", "承诺书", "意向书",
)

CONTRACT_AGREEMENT_TITLE_MARKERS = (
    "保理协议", "付款协议", "施工协议", "采购协议", "服务协议",
    "租赁协议", "借款协议", "合作协议", "购销协议", "承包协议",
    "分包协议", "委托协议", "框架协议", "买卖合同",
)

CONTRACT_BODY_STRUCTURE_FEATURES = (
    "甲方", "乙方", "丙方", "合同编号", "协议编号",
    "签订日期", "签署日期", "生效日期", "权利义务", "违约责任",
    "合同金额", "鉴于", "双方约定", "合同价款", "协议价款",
)


def infer_title_region_from_text(text: str, top_ratio: float = 0.2) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return "\n".join(lines[: max(1, int(len(lines) * top_ratio))])
    return text[: max(50, int(len(text) * top_ratio))]


def _compact_title(title: str) -> str:
    return re.sub(r"\s+", "", str(title or ""))


def match_contract_title(title_region: str) -> Tuple[bool, str]:
    title = _compact_title(title_region)
    if not title:
        return False, ""
    for marker in NON_CONTRACT_AGREEMENT_TITLE_MARKERS:
        if marker in title:
            return False, marker
    if "合同" in title:
        return True, "合同"
    for marker in CONTRACT_AGREEMENT_TITLE_MARKERS:
        if marker in title:
            return True, marker
    if "协议" in title:
        exclude = ("报告", "通知书", "确认函", "结算单", "台账", "明细表", "回单", "许可证", "承诺")
        if any(w in title for w in exclude):
            return False, ""
        return True, "协议"
    return False, ""


def has_contract_body_features(text: str, min_features: int = 2) -> Tuple[bool, List[str]]:
    if not text:
        return False, []
    compact = re.sub(r"\s+", "", str(text))
    hits = [f for f in CONTRACT_BODY_STRUCTURE_FEATURES if f in compact or f in text]
    return len(hits) >= min_features, hits


def is_business_license_title_positioned(ocr_items: list, full_text: str = None) -> bool:
    if hasattr(ocr_module, "is_business_license_title_positioned"):
        return ocr_module.is_business_license_title_positioned(ocr_items or [], full_text=full_text)
    return False


def detect_document_type_by_title(
    text: str,
    title_region: Optional[str] = None,
    ocr_items: Optional[list] = None,
) -> str:
    """根据标题区域识别文档类型（标题优先原则）。"""
    if not text and not title_region:
        return ""

    title = (title_region or infer_title_region_from_text(text or "")).lower()

    for keyword, doc_type in TITLE_RULES:
        if keyword not in title:
            continue
        if doc_type == "营业执照":
            if not is_business_license_title_positioned(ocr_items or [], full_text=text):
                continue
        if doc_type == "发票":
            ledger_features = ["序号", "支付时间", "收款单位", "支付金额", "合计"]
            if sum(1 for f in ledger_features if f in (text or "")) >= 4:
                continue
        return doc_type

    contract_hit, _ = match_contract_title(title)
    if contract_hit:
        return "合同"

    return ""
