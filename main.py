"""主应用程序 - Document Agent 系统入口"""
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from agent.document_agent import create_document_agent
from config import get_api_key, get_base_url, get_model, mask_key

_DEFAULT_OPENAI_KEY = None
api_key = get_api_key()
if not api_key:
    print("警告: 未设置 OPENAI_API_KEY，请在 .env 或环境变量中配置")
base_url = get_base_url()
model = get_model()
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时连接 MCP，关闭时释放资源。"""
    global agent
    print(f"LLM: model={model}, base_url={base_url}, key={mask_key(api_key)}")
    agent = create_document_agent(
        api_key=api_key,
        base_url=base_url,
        model=model,
        auto_connect_mcp=True,
    )
    yield
    if agent:
        agent.close()


app = FastAPI(
    title="Document Intelligence Agent",
    description="基于 LangChain + MCP 的文档智能处理系统",
    version="1.2.0",
    lifespan=lifespan,
)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

CLIENT_DIR = Path(__file__).parent / "client"
if CLIENT_DIR.exists():
    app.mount("/client", StaticFiles(directory=str(CLIENT_DIR)), name="client")


@app.get("/")
async def root():
    """Web 客户端入口"""
    if (CLIENT_DIR / "index.html").exists():
        return RedirectResponse(url="/client/index.html")
    return RedirectResponse(url="/docs")


@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    task: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """上传并处理文档（兼容旧接口）"""
    try:
        file_path = (UPLOAD_DIR / file.filename).resolve()
        content = await file.read()
        file_path.write_bytes(content)

        sid = session_id or str(uuid.uuid4())
        result = agent.run(str(file_path), task, session_id=sid)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/chat")
async def chat(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """多轮对话：可选上传文件触发完整文档处理流程。"""
    try:
        sid = session_id or str(uuid.uuid4())
        doc_path = None

        if file and file.filename:
            file_path = (UPLOAD_DIR / file.filename).resolve()
            content = await file.read()
            file_path.write_bytes(content)
            doc_path = str(file_path)

        result = agent.chat(message=message, session_id=sid, doc_path=doc_path)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e), "success": False}, status_code=500)


@app.get("/chat/history")
async def chat_history(
    session_id: str = Query(...),
    limit: int = Query(50, ge=1, le=200),
):
    """获取指定会话的对话历史。"""
    try:
        messages = agent.get_chat_history(session_id, limit=limit)
        return JSONResponse(content={"session_id": session_id, "messages": messages})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/chat/clear")
async def clear_chat(session_id: str = Form(...)):
    """清空指定会话历史。"""
    try:
        agent.clear_session(session_id)
        return JSONResponse(content={"session_id": session_id, "cleared": True})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/documents")
async def list_uploaded_documents():
    """列出 uploads 目录中的历史文件。"""
    try:
        files = []
        for path in UPLOAD_DIR.iterdir():
            if not path.is_file():
                continue
            stat = path.stat()
            files.append({
                "name": path.name,
                "path": str(path.resolve()),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        files.sort(key=lambda x: x["modified"], reverse=True)
        return JSONResponse(content={"files": files, "count": len(files)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/knowledge/list")
async def list_knowledge(
    limit: int = Query(100, ge=1, le=500),
    preview_len: int = Query(200, ge=50, le=1000),
):
    """列出 ChromaDB 知识库中的文档。"""
    try:
        docs = agent.rag_memory.list_documents(limit=limit, text_preview_len=preview_len)
        return JSONResponse(content={"documents": docs, "count": len(docs)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/status")
async def get_agent_status():
    """获取 Agent 状态"""
    try:
        status = agent.get_agent_status()
        status["llm"] = {"model": model, "base_url": base_url}
        return JSONResponse(content=status, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/health")
async def health_check():
    """健康检查"""
    return JSONResponse(
        content={"status": "healthy", "agent": "ready" if agent else "initializing"},
        status_code=200,
    )


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """启动服务器"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
