"""文档类型知识图谱 — 25 类文档规则、关系与检索。"""
import json
from typing import Any, Dict, List, Optional

from knowledge.rules_data import DOCUMENT_RELATIONS, DOCUMENT_VALIDATORS
from knowledge.title_detection import detect_document_type_by_title


class DocumentKnowledgeGraph:
    """
    结构化知识图谱（非图数据库，内存邻接表）。

    节点：文档类型
    边：DOCUMENT_RELATIONS（如 合同 → 补充协议）
    属性：keywords / pass_score / title_rules
    """

    def __init__(self):
        self._types = DOCUMENT_VALIDATORS
        self._relations = DOCUMENT_RELATIONS
        self._adjacency: Dict[str, List[Dict[str, str]]] = {}
        for src, dst, rel in self._relations:
            self._adjacency.setdefault(src, []).append({"target": dst, "relation": rel})

    def list_document_types(self) -> List[str]:
        return list(self._types.keys())

    def get_type_info(self, doc_type: str) -> Optional[Dict[str, Any]]:
        if doc_type not in self._types:
            return None
        rules = self._types[doc_type]
        return {
            "doc_type": doc_type,
            "pass_score": rules.get("pass_score"),
            "keywords": list(rules.get("keywords", {}).keys()),
            "exclude_keywords": rules.get("exclude_keywords", []),
            "required_keywords": rules.get("required_keywords", []),
            "related_types": self.get_related_types(doc_type),
        }

    def get_related_types(self, doc_type: str) -> List[Dict[str, str]]:
        return self._adjacency.get(doc_type, [])

    def search_types(self, query: str) -> List[Dict[str, Any]]:
        """按关键词搜索匹配的文档类型（图谱检索）。"""
        query = (query or "").strip()
        if not query:
            return []

        results = []
        for doc_type, rules in self._types.items():
            keywords = rules.get("keywords", {})
            hits = [kw for kw in keywords if kw in query or query in kw]
            if doc_type in query:
                hits.insert(0, doc_type)
            if hits:
                results.append({
                    "doc_type": doc_type,
                    "matched_keywords": hits[:5],
                    "pass_score": rules.get("pass_score"),
                    "top_keyword_weights": {
                        kw: keywords[kw] for kw in hits[:3] if kw in keywords
                    },
                })
        results.sort(key=lambda x: len(x["matched_keywords"]), reverse=True)
        return results[:5]

    def classify_text(
        self,
        text: str,
        title_region: Optional[str] = None,
        ocr_items: Optional[list] = None,
    ) -> Dict[str, Any]:
        """基于标题规则推断文档类型。"""
        detected = detect_document_type_by_title(text, title_region, ocr_items)
        if not detected:
            return {"predicted_type": "", "method": "title_detection", "confidence": 0.0}

        rules = self._types.get(detected, {})
        return {
            "predicted_type": detected,
            "method": "title_detection",
            "confidence": 1.0,
            "type_info": {
                "pass_score": rules.get("pass_score"),
                "keywords": list(rules.get("keywords", {}).keys())[:8],
            },
        }

    def to_json_summary(self) -> str:
        """导出图谱摘要供 Agent 阅读。"""
        summary = {
            "document_types": self.list_document_types(),
            "total_types": len(self._types),
            "relations_sample": self._relations[:8],
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)
