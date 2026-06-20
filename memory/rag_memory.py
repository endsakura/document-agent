"""RAG 知识检索与 Memory 存储模块 - 使用 ChromaDB 向量数据库"""
import json
import chromadb
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


def _sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Union[str, int, float, bool]]:
    """
    ChromaDB metadata 仅支持 str/int/float/bool。
    列表、字典等复杂类型转为 JSON 字符串；空值跳过。
    """
    if not metadata:
        return {}

    clean: Dict[str, Union[str, int, float, bool]] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, list):
            if not value:
                continue
            clean[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, dict):
            clean[key] = json.dumps(value, ensure_ascii=False)
        else:
            clean[key] = str(value)
    return clean


class RAGMemory:
    def __init__(self, collection_name: str = "documents", persist_dir: str = "./chroma_db"):
        """初始化 RAG Memory"""
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.collection_name = collection_name

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> bool:
        """添加文档到向量库（同 doc_id 已存在则更新）"""
        try:
            meta = _sanitize_metadata(metadata or {})
            meta["added_at"] = datetime.now().isoformat()

            # upsert：避免 Agent 重试时 doc_id 冲突
            self.collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[meta],
            )
            return True
        except Exception as e:
            print(f"Error adding document: {e}")
            return False

    def retrieve_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """检索相似文档"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            documents = []
            if results["ids"] and len(results["ids"]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    documents.append({
                        "id": doc_id,
                        "text": results["documents"][0][i]
                    })

            return documents
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def get_all_documents(self) -> List[Dict]:
        """获取所有文档"""
        try:
            results = self.collection.get()
            documents = []
            if results["ids"]:
                metadatas = results.get("metadatas") or []
                for i, doc_id in enumerate(results["ids"]):
                    documents.append({
                        "id": doc_id,
                        "text": results["documents"][i],
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                    })
            return documents
        except Exception as e:
            print(f"Error getting all documents: {e}")
            return []

    def list_documents(
        self,
        limit: int = 100,
        text_preview_len: int = 200,
    ) -> List[Dict]:
        """列出知识库文档（含 metadata 与文本预览）。"""
        try:
            results = self.collection.get()
            documents: List[Dict] = []
            if not results["ids"]:
                return documents

            metadatas = results.get("metadatas") or []
            for i, doc_id in enumerate(results["ids"][:limit]):
                text = results["documents"][i] or ""
                meta = metadatas[i] if i < len(metadatas) else {}
                preview = text[:text_preview_len]
                if len(text) > text_preview_len:
                    preview += "…"
                documents.append({
                    "id": doc_id,
                    "text_preview": preview,
                    "text_length": len(text),
                    "doc_type": meta.get("doc_type", ""),
                    "added_at": meta.get("added_at", ""),
                    "metadata": meta,
                })
            return documents
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []


class ConversationMemory:
    """对话历史记录"""

    def __init__(self):
        self.messages = []

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加消息"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)

    def get_recent_messages(self, limit: int = 10) -> List[Dict]:
        """获取最近的消息"""
        return self.messages[-limit:]

    def clear(self):
        """清空对话历史"""
        self.messages = []


def create_rag_memory(collection_name: str = "documents") -> RAGMemory:
    """工厂函数"""
    return RAGMemory(collection_name)


def create_conversation_memory() -> ConversationMemory:
    """工厂函数"""
    return ConversationMemory()
