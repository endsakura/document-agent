"""文档分类器 — 结合 YOLO 提示 + OCR 文本 + 25 类知识图谱规则。"""
import re
from typing import Any, Dict, List, Optional

from knowledge.rules_data import DOCUMENT_VALIDATORS
from knowledge.title_detection import (
    detect_document_type_by_title,
    infer_title_region_from_text,
)

# YOLO 检测类别 → 知识图谱文档类型（可按 best.pt 训练标签扩展）
YOLO_CLASS_TO_DOC_TYPE = {
    "id_card": "法人身份证",
    "idcard": "法人身份证",
    "business_license": "营业执照",
    "license": "营业执照",
    "contract": "合同",
    "invoice": "发票",
    "report": "财务报表",
    "financial_report": "财务报表",
    "bid": "中标通知书",
    "bid_notice": "中标通知书",
    "company_profile": "公司简介",
    "profile": "公司简介",
    "charter": "公司章程",
    "certificate": "建筑资质证书",
    "bank_receipt": "银行回单",
    "audit_report": "审计报告",
}


def _normalize_yolo_classes(yolo_classes: Optional[List[str]]) -> List[str]:
    if not yolo_classes:
        return []
    normalized = []
    for cls in yolo_classes:
        if not cls or cls.lower() == "other":
            continue
        normalized.append(cls.lower())
    return normalized


def _yolo_doc_type_hints(yolo_classes: Optional[List[str]]) -> List[str]:
    hints = []
    for cls in _normalize_yolo_classes(yolo_classes):
        mapped = YOLO_CLASS_TO_DOC_TYPE.get(cls)
        if mapped and mapped not in hints:
            hints.append(mapped)
    return hints


class DocumentClassifier:
    """基于知识图谱规则，将文档归入 25 类中的唯一类别。"""

    VALIDATORS = DOCUMENT_VALIDATORS

    def classify_from_text(
        self,
        text: str,
        title_region: str = "",
        ocr_items: Optional[list] = None,
        yolo_classes: Optional[List[str]] = None,
        file_name: str = "",
    ) -> Dict[str, Any]:
        """
        根据 OCR 文本 + 知识图谱，输出唯一 final_category。
        YOLO 仅作参考；当 YOLO 为 other 或未检出时，完全依赖知识库规则。
        """
        text = (text or "").strip()
        if not text:
            return {"success": False, "error": "OCR 文本为空，无法分类"}

        title_region = title_region or infer_title_region_from_text(text)
        yolo_hints = _yolo_doc_type_hints(yolo_classes)

        # 1) 标题优先
        title_type = detect_document_type_by_title(text, title_region, ocr_items or [])

        # 2) 全类型打分
        rankings: List[Dict[str, Any]] = []
        for doc_type in self.VALIDATORS:
            scored = self._score_type(text, doc_type, title_region, ocr_items or [])
            bonus = 15 if doc_type in yolo_hints else 0
            if title_type == doc_type:
                bonus += 25
            total = scored["score"] + bonus
            rankings.append({
                "doc_type": doc_type,
                "score": scored["score"],
                "bonus": bonus,
                "total": total,
                "confidence": scored["confidence"],
                "is_valid": scored["is_valid"],
                "matched_keywords": scored["matched_keywords"][:6],
            })

        rankings.sort(key=lambda x: x["total"], reverse=True)

        # 3) 确定唯一类别
        final_category, method = self._pick_final_category(
            title_type, rankings, yolo_hints
        )

        validation = self._score_type(text, final_category, title_region, ocr_items or [])

        return {
            "success": True,
            "final_category": final_category,
            "method": method,
            "confidence": validation["confidence"],
            "is_valid": validation["is_valid"],
            "title_region": title_region[:200],
            "yolo_hints": yolo_hints,
            "yolo_insufficient": not yolo_hints,
            "top_candidates": rankings[:5],
            "matched_keywords": validation["matched_keywords"][:8],
            "file_name": file_name,
            "analysis_hint": self._analysis_hint(final_category),
        }

    def _pick_final_category(
        self,
        title_type: str,
        rankings: List[Dict[str, Any]],
        yolo_hints: List[str],
    ) -> tuple:
        if title_type:
            return title_type, "title_detection"

        valid = [r for r in rankings if r["is_valid"]]
        if valid:
            return valid[0]["doc_type"], "knowledge_graph_valid"

        if yolo_hints:
            for hint in yolo_hints:
                for r in rankings:
                    if r["doc_type"] == hint:
                        return hint, "yolo_hint+knowledge_graph"
            return yolo_hints[0], "yolo_hint"

        if rankings:
            return rankings[0]["doc_type"], "knowledge_graph_scoring"

        return "未知", "none"

    def _score_type(
        self,
        text: str,
        doc_type: str,
        title_region: str,
        ocr_items: list,
    ) -> Dict[str, Any]:
        from knowledge.document_validator import DocumentValidator

        validator = DocumentValidator()
        validator._title_region = title_region
        validator._ocr_items = ocr_items
        return validator._score_document(text, doc_type)

    def _analysis_hint(self, doc_type: str) -> str:
        hints = {
            "法人身份证": "提取姓名、身份证号、住址、签发机关、有效期",
            "营业执照": "提取企业名称、统一社会信用代码、法定代表人、经营范围",
            "公司简介": "总结公司基本情况、治理结构、业务与发展规划",
            "财务报表": "分析资产、负债、收入、利润等核心指标",
            "中标通知书": "提取项目名称、中标人、中标金额、招标人",
            "合同": "提取甲乙方、合同金额、签订日期、主要条款",
        }
        return hints.get(doc_type, f"按「{doc_type}」文档要点进行结构化分析")
