"""MCP 客户端 - 通过 stdio 与 mcp_server 通信"""
import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent


class MCPClient:
    """持久化 MCP 客户端，在独立线程中维护 asyncio 事件循环。"""

    def __init__(self, server_script: Optional[Path] = None):
        self._server_script = server_script or Path(__file__).parent / "mcp_server.py"
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._stdio_ctx = None
        self._session_ctx = None
        self._ready = threading.Event()
        self._connect_error: Optional[Exception] = None

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
            self._ready.set()
            self._loop.run_forever()
        except Exception as exc:
            self._connect_error = exc
            self._ready.set()
        finally:
            try:
                if self._loop.is_running():
                    self._loop.run_until_complete(self._disconnect())
            except Exception:
                pass
            self._loop.close()

    async def _connect(self) -> None:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(self._server_script)],
            cwd=str(self._server_script.parent),
        )
        self._stdio_ctx = stdio_client(server_params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()

    async def _disconnect(self) -> None:
        if self._session_ctx:
            await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None
            self._session = None
        if self._stdio_ctx:
            await self._stdio_ctx.__aexit__(None, None, None)
            self._stdio_ctx = None

    def connect(self, timeout: float = 120) -> None:
        """启动 MCP 子进程并建立连接。"""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError("MCP 服务器连接超时")
        if self._connect_error:
            raise self._connect_error

    def disconnect(self) -> None:
        """关闭 MCP 连接。"""
        if not self._loop or not self._loop.is_running():
            return
        future = asyncio.run_coroutine_threadsafe(self._disconnect(), self._loop)
        try:
            future.result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    def _run_async(self, coro, timeout: float = 300):
        if not self._loop or not self._session:
            raise RuntimeError("MCP 客户端未连接，请先调用 connect()")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    @staticmethod
    def _extract_text(result) -> str:
        parts = []
        for block in result.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts) if parts else ""

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出 MCP 服务器提供的所有工具。"""
        async def _list():
            response = await self._session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                }
                for tool in response.tools
            ]

        return self._run_async(_list())

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """同步调用 MCP 工具。"""
        async def _call():
            result = await self._session.call_tool(name, arguments=arguments or {})
            if getattr(result, "isError", False):
                return f"工具错误: {self._extract_text(result)}"
            return self._extract_text(result)

        return self._run_async(_call())

    def call_tool_json(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """调用工具并尝试解析 JSON 结果。"""
        text = self.call_tool(name, arguments)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
