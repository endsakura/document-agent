"""MCP 服务器 - 暴露 4 个统一工具：ocr_tool / yolo_tool"""
from mcp.server.fastmcp import FastMCP

from tools.router import run_ocr, run_yolo

mcp = FastMCP("document-agent-tools")


@mcp.tool()
def ocr_tool(input: str) -> str:
    """
    统一 OCR 工具，内部自动判断处理策略，无需选择具体 OCR 方法。

    传入文件路径或 JSON：
    - 纯路径：自动识别 PDF/图片并提取文本
    - JSON 示例：{"path": "/data/doc.jpg", "task": "提取标题"}
    - 可选 mode：auto（默认）| text | pdf | title | items
    - PDF 可选 max_pages；标题模式可选 top_ratio
    """
    return run_ocr(input)


@mcp.tool()
def yolo_tool(input: str) -> str:
    """
    统一视觉检测工具，识别文档类型和关键对象。支持图片和 PDF。

    传入文件路径；PDF 会先渲染为页图再 YOLO 检测（默认第 1 页）。
    多页 PDF 可传 JSON：{"path": "...", "max_pages": 3}
    """
    return run_yolo(input)


if __name__ == "__main__":
    mcp.run(transport="stdio")
