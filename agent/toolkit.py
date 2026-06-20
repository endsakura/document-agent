"""LangChain 工具注册层 — 4 个统一工具，供 Agent 调度。"""
from typing import List

from langchain.agents import Tool

from mcp_client import MCPClient
from knowledge.service import KnowledgeService
from memory.rag_memory import RAGMemory

# 注意：Tool description 会嵌入 LangChain ReAct 模板，不能含未转义的花括号 {{ }}

_OCR_TOOL_DESC = (
    "统一 OCR 工具（PaddleOCR）。传入文件路径；"
    "自动识别 PDF 或图片并提取文本。"
    "需要标题提取时可在 JSON 中附加 task 字段。"
)

_YOLO_TOOL_DESC = (
    "统一视觉检测工具。支持图片和 PDF，识别文档类型和关键对象。"
    "PDF 会先转页图再 YOLO 检测，默认第 1 页；"
    "多页可在 JSON 中设 max_pages。"
)

_KNOWLEDGE_SEARCH_DESC = (
    "知识检索与文档分类（向量库 + 25类知识图谱）。"
    "OCR 后必须调用 classify_document 自主分类，确定唯一 final_category。"
    "JSON action：classify_document（path + yolo_classes）、validate、list_types、"
    "type_info、graph_search。"
    "YOLO 为 other 时仍可用 classify_document 基于知识库分类。"
)

_KNOWLEDGE_STORE_DESC = (
    "存入向量知识库。传入 JSON 字符串，"
    "必填字段 doc_id、text；可选 metadata 含 doc_type。"
)


def build_langchain_tools(
    mcp_client: MCPClient,
    rag_memory: RAGMemory,
) -> List[Tool]:
    """
    构建 LangChain Agent 所需的 4 个工具：

        LangChain Agent
        ├── ocr_tool          (MCP → PaddleOCR)
        ├── yolo_tool         (MCP)
        ├── knowledge_search  (Vector DB + 知识图谱 + 验证器)
        └── knowledge_store   (Vector DB)
    """
    knowledge = KnowledgeService(rag_memory)

    def ocr_tool(raw_input: str) -> str:
        return mcp_client.call_tool("ocr_tool", {"input": raw_input})

    def yolo_tool(raw_input: str) -> str:
        return mcp_client.call_tool("yolo_tool", {"input": raw_input})

    def knowledge_search(query: str) -> str:
        return knowledge.search(query)

    def knowledge_store(raw_input: str) -> str:
        return knowledge.store(raw_input)

    return [
        Tool(name="ocr_tool", func=ocr_tool, description=_OCR_TOOL_DESC),
        Tool(name="yolo_tool", func=yolo_tool, description=_YOLO_TOOL_DESC),
        Tool(
            name="knowledge_search",
            func=knowledge_search,
            description=_KNOWLEDGE_SEARCH_DESC,
        ),
        Tool(
            name="knowledge_store",
            func=knowledge_store,
            description=_KNOWLEDGE_STORE_DESC,
        ),
    ]
