"""
Document Agent — 薄编排层。
所有工具调用由 LangChain Agent 统一调度。
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent

from mcp_client import MCPClient
from memory.rag_memory import create_rag_memory
from memory.session_store import SessionStore
from agent.toolkit import build_langchain_tools

_RUN_PROMPT = """\
处理文档: {doc_path}
任务: {task}

你必须按以下顺序处理，不可跳过分类步骤：

【步骤1】yolo_tool
- 传入文档路径，获取视觉检测提示（可能仅为 other，属正常情况）

【步骤2】ocr_tool
- 传入文档路径，提取全文（必须执行）

【步骤3】knowledge_search — 自主分类（必须执行）
- 当 YOLO 未给出明确类别（如 other）或置信度低时，必须依赖知识库分类
- 调用 classify_document：传入 path 与 yolo_classes（从步骤1取 class 名）
- 从返回结果中确定唯一 final_category（25类之一）
- 记录 confidence、top_candidates

【步骤4】knowledge_search — 校验（建议）
- 用 validate 校验 final_category 是否匹配，path + doc_type

【步骤5】分析文档
- 根据 final_category 做结构化分析（参考 analysis_hint）
- 输出该类型应关注的关键字段与结论

【步骤6】knowledge_store
- 存入知识库，metadata.doc_type 必须等于 final_category

注意：
- 每个文档只能归入一个 final_category
- YOLO 只是辅助，分类以知识图谱 classify_document 为准
- 可用 list_types 查看支持的 25 类文档
"""

_CHAT_PROMPT = """\
你是 Document Agent 文档智能助手。请结合对话历史与用户问题作答。

若问题涉及已入库文档，请用 knowledge_search 检索相关内容后再回答。
若用户需要处理新文档，请提示其在对话中上传文件。
回答请简洁、结构化，使用中文。

【对话历史】
{history}

【当前问题】
{message}
"""


def _format_history(messages: List[Dict], limit: int = 8) -> str:
    lines: List[str] = []
    for msg in messages[-limit:]:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) if lines else "（无历史）"


class DocumentAgent:
    """文档智能 Agent — run() 处理文档，chat() 多轮对话。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        mcp_client: Optional[MCPClient] = None,
        auto_connect_mcp: bool = True,
    ):
        self.mcp_client = mcp_client or MCPClient()
        self.rag_memory = create_rag_memory()
        self.session_store = SessionStore()
        self.execution_history: List[Dict] = []

        if auto_connect_mcp:
            self.mcp_client.connect()

        llm_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "model": model,
            "temperature": 0.3,
        }
        if base_url:
            llm_kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**llm_kwargs)
        self.tools = build_langchain_tools(self.mcp_client, self.rag_memory)
        self.agent_executor = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True,
        )

    def run(self, doc_path: str, task: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        task = task or "识别文档类型、分类并分析内容"
        session_id = session_id or "default"
        memory = self.session_store.get(session_id)
        memory.add_message(
            "user",
            f"{task} | {doc_path}",
            metadata={"doc_path": doc_path, "mode": "process"},
        )

        result: Dict[str, Any] = {
            "document": doc_path,
            "task": task,
            "session_id": session_id,
            "mode": "process",
            "timestamp": datetime.now().isoformat(),
        }

        try:
            prompt = _RUN_PROMPT.format(doc_path=doc_path, task=task)
            response = self.agent_executor.invoke({"input": prompt})
            output = response.get("output", "")
            result["final_result"] = output
            result["reply"] = output
            result["success"] = True
            memory.add_message("assistant", output, metadata={"mode": "process"})
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            memory.add_message("assistant", f"处理失败: {e}", metadata={"mode": "process"})

        self.execution_history.append(result)
        return result

    def chat(
        self,
        message: str,
        session_id: str,
        doc_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """多轮对话：带 doc_path 时走完整文档流程，否则基于历史与知识库问答。"""
        if doc_path:
            return self.run(doc_path, task=message, session_id=session_id)

        memory = self.session_store.get(session_id)
        memory.add_message("user", message)

        result: Dict[str, Any] = {
            "session_id": session_id,
            "mode": "chat",
            "timestamp": datetime.now().isoformat(),
        }

        try:
            history = _format_history(memory.messages[:-1])
            prompt = _CHAT_PROMPT.format(history=history, message=message)
            response = self.agent_executor.invoke({"input": prompt})
            output = response.get("output", "")
            result["reply"] = output
            result["success"] = True
            memory.add_message("assistant", output, metadata={"mode": "chat"})
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            memory.add_message("assistant", f"回答失败: {e}", metadata={"mode": "chat"})

        return result

    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        memory = self.session_store.get(session_id)
        return memory.get_recent_messages(limit)

    def clear_session(self, session_id: str) -> None:
        self.session_store.clear(session_id)

    def get_agent_status(self) -> Dict[str, Any]:
        return {
            "status": "ready",
            "tools": [t.name for t in self.tools],
            "memory_size": len(self.rag_memory.get_all_documents()),
            "sessions": self.session_store.count(),
            "executions": len(self.execution_history),
        }

    def close(self) -> None:
        self.mcp_client.disconnect()


def create_document_agent(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    base_url: Optional[str] = None,
    mcp_client: Optional[MCPClient] = None,
    auto_connect_mcp: bool = True,
) -> DocumentAgent:
    return DocumentAgent(
        api_key=api_key,
        model=model,
        base_url=base_url,
        mcp_client=mcp_client,
        auto_connect_mcp=auto_connect_mcp,
    )
