"""知识服务 — 统一向量 RAG + 文档知识图谱 + 验证器。"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

from knowledge.document_validator import DocumentValidator
from knowledge.graph import DocumentKnowledgeGraph
from knowledge.classifier import DocumentClassifier
from memory.rag_memory import RAGMemory


class KnowledgeService:
    """
    知识层统一入口。

    Vector DB  : 语义检索历史文档片段
    知识图谱    : 25 类文档规则、类型关系
    验证器      : OCR + 规则校验 / 自动分类
    """

    def __init__(self, rag_memory: RAGMemory):
        self.rag = rag_memory
        self.graph = DocumentKnowledgeGraph()
        self.classifier = DocumentClassifier()
        self._validator: Optional[DocumentValidator] = None

    @property
    def validator(self) -> DocumentValidator:
        if self._validator is None:
            self._validator = DocumentValidator()
        return self._validator

    def search(self, raw_input: str) -> str:
        """
        知识检索（向量 + 图谱）。

        纯文本       → 向量检索 + 图谱关键词匹配
        JSON action  → 结构化查询：

          {"action": "classify", "path": "/data/doc.jpg"}
          {"action": "classify_document", "path": "...", "text": "OCR全文", "yolo_classes": ["other"]}
          {"action": "classify_text", "text": "...", "title": "..."}
          {"action": "validate", "path": "/data/doc.jpg", "doc_type": "营业执照"}
          {"action": "graph_summary"}
          {"action": "list_types"}
        """
        raw_input = (raw_input or "").strip()
        if raw_input.startswith("{"):
            try:
                payload = json.loads(raw_input)
                return self._search_structured(payload)
            except json.JSONDecodeError:
                pass

        return self._search_hybrid(raw_input)

    def store(self, raw_input: str) -> str:
        """
        存入知识库。

        JSON：{"doc_id": "...", "text": "...", "metadata": {"doc_type": "合同"}}
        metadata.doc_type 会同步写入图谱索引（向量库 metadata）
        """
        try:
            payload = json.loads(raw_input)
            doc_id = payload["doc_id"]
            text = payload["text"]
            metadata = payload.get("metadata") or {}
        except (json.JSONDecodeError, KeyError):
            return '格式错误：{"doc_id": "...", "text": "...", "metadata": {}}'

        if "doc_type" in metadata and metadata["doc_type"] in self.graph.list_document_types():
            type_info = self.graph.get_type_info(metadata["doc_type"])
            metadata["knowledge_graph"] = True
            related = type_info.get("related_types", []) if type_info else []
            if related:
                # 存为 JSON 字符串，避免 ChromaDB 拒绝空列表/嵌套 dict
                metadata["related_types"] = json.dumps(related, ensure_ascii=False)

        ok = self.rag.add_document(doc_id, text, metadata)
        if ok:
            return f"已存入知识库（doc_id={doc_id}）"
        return "存入失败：请检查 doc_id、text 是否有效，或查看服务端日志"

    def _search_hybrid(self, query: str) -> str:
        parts = []

        vector_docs = self.rag.retrieve_similar(query, n_results=3)
        if vector_docs:
            parts.append("【向量检索】")
            parts.extend(doc["text"] for doc in vector_docs)

        graph_hits = self.graph.search_types(query)
        if graph_hits:
            parts.append("【知识图谱】")
            for hit in graph_hits:
                parts.append(
                    f"- {hit['doc_type']}（匹配: {', '.join(hit['matched_keywords'][:3])}）"
                )

        return "\n".join(parts) if parts else "未找到相关信息"

    def _search_structured(self, payload: Dict[str, Any]) -> str:
        action = payload.get("action", "search")

        if action == "list_types":
            return json.dumps(self.graph.list_document_types(), ensure_ascii=False)

        if action == "classify_document":
            return self._classify_document(payload)

        if action == "type_info":
            doc_type = payload.get("doc_type", "")
            info = self.graph.get_type_info(doc_type)
            return json.dumps(info or {"error": f"未知类型: {doc_type}"}, ensure_ascii=False, indent=2)

        if action == "graph_search":
            hits = self.graph.search_types(payload.get("query", ""))
            return json.dumps(hits, ensure_ascii=False, indent=2)

        if action == "graph_summary":
            return self.graph.to_json_summary()

        if action == "classify":
            path = payload.get("path", "")
            result = self.validator.classify(path)
            return json.dumps(result, ensure_ascii=False, indent=2)

        if action == "classify_text":
            result = self.graph.classify_text(
                payload.get("text", ""),
                title_region=payload.get("title"),
                ocr_items=payload.get("ocr_items"),
            )
            return json.dumps(result, ensure_ascii=False, indent=2)

        if action == "validate":
            path = payload.get("path", "")
            doc_type = payload.get("doc_type", "")
            result = self.validator.validate(path, doc_type)
            return json.dumps(result, ensure_ascii=False, indent=2)

        if action == "search":
            return self._search_hybrid(payload.get("query", ""))

        return json.dumps({"error": f"未知 action: {action}"}, ensure_ascii=False)

    def _classify_document(self, payload: Dict[str, Any]) -> str:
        """
        OCR 后自主分类：结合 YOLO 提示 + 知识图谱，输出唯一 final_category。
        优先传 text（OCR 结果）；也可传 path（会重新 OCR）。
        """
        yolo_classes = payload.get("yolo_classes") or payload.get("yolo_hint") or []
        if isinstance(yolo_classes, str):
            yolo_classes = [yolo_classes]

        text = payload.get("text", "")
        title = payload.get("title", "")
        path = payload.get("path", "")
        file_name = payload.get("file_name", "")
        ocr_items = payload.get("ocr_items")

        if not text and path:
            result = self.validator.classify(path)
            if not result.get("success"):
                return json.dumps(result, ensure_ascii=False, indent=2)
            text = self.validator._last_extracted_text
            title = title or self.validator._title_region
            file_name = file_name or result.get("file", "")
            ocr_items = ocr_items or self.validator._ocr_items
        elif not text:
            return json.dumps(
                {"success": False, "error": "classify_document 需要 text 或 path"},
                ensure_ascii=False,
            )
        elif not file_name and path:
            file_name = Path(path).name

        classified = self.classifier.classify_from_text(
            text=text,
            title_region=title,
            ocr_items=ocr_items,
            yolo_classes=yolo_classes,
            file_name=file_name,
        )
        return json.dumps(classified, ensure_ascii=False, indent=2)
