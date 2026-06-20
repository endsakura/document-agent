"""统一工具路由 — OCR / YOLO 内部自动判断执行策略。"""
import json
from pathlib import Path
from typing import Any, Dict

from tools.ocr import (
    collect_ocr_items,
    extract_text_and_title_from_image,
    extract_text_from_image,
    extract_text_from_pdf,
)
from tools.path_utils import resolve_file_path

PDF_EXTENSIONS = {".pdf"}

_yolo_detector = None


def _get_yolo_detector():
    global _yolo_detector
    if _yolo_detector is None:
        from tools.yolo_tool import create_yolo_tool
        _yolo_detector = create_yolo_tool()
    return _yolo_detector


def parse_tool_input(raw_input: str) -> Dict[str, Any]:
    raw_input = (raw_input or "").strip()
    if not raw_input:
        return {}
    if raw_input.startswith("{"):
        try:
            return json.loads(raw_input)
        except json.JSONDecodeError:
            pass
    return {"path": raw_input}


def _extract_path(payload: Dict[str, Any]) -> str:
    for key in ("path", "file_path", "image_path", "pdf_path"):
        if payload.get(key):
            return str(resolve_file_path(str(payload[key])))
    raise ValueError("缺少文件路径，请传入 path 或直接传路径字符串")


def resolve_ocr_mode(path: str, mode: str = "auto", task: str = "") -> str:
    if mode and mode != "auto":
        return mode

    ext = Path(path).suffix.lower()
    if ext in PDF_EXTENSIONS:
        return "pdf"

    hint = (task or "").lower()
    if any(k in hint for k in ("标题", "title", "题目", "抬头")):
        return "title"
    if any(k in hint for k in ("位置", "bbox", "坐标", "区域", "文本块", "layout")):
        return "items"

    return "text"


def run_ocr(raw_input: str) -> str:
    try:
        payload = parse_tool_input(raw_input)
        path = _extract_path(payload)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"mode": "error", "text": str(e)}, ensure_ascii=False)

    mode = resolve_ocr_mode(
        path,
        mode=payload.get("mode", "auto"),
        task=payload.get("task", ""),
    )

    if mode == "pdf":
        max_pages = payload.get("max_pages")
        pages = extract_text_from_pdf(path, max_pages=max_pages)
        text = "\n---PAGE_BREAK---\n".join(pages)
        return json.dumps(
            {
                "mode": "pdf",
                "path": path,
                "pages": len(pages),
                "text": text or "未提取到文本",
            },
            ensure_ascii=False,
        )

    if mode == "title":
        top_ratio = float(payload.get("top_ratio", 0.2))
        full_text, title_text, items = extract_text_and_title_from_image(path, top_ratio)
        return json.dumps(
            {
                "mode": "title",
                "path": path,
                "full_text": full_text,
                "title_text": title_text,
                "items_count": len(items),
            },
            ensure_ascii=False,
        )

    if mode == "items":
        items = collect_ocr_items(path)
        return json.dumps({"mode": "items", "path": path, "items": items}, ensure_ascii=False)

    text = extract_text_from_image(path)
    return json.dumps(
        {"mode": "text", "path": path, "text": text or "未提取到文本"},
        ensure_ascii=False,
    )


def run_yolo(raw_input: str) -> str:
    try:
        payload = parse_tool_input(raw_input)
        path = _extract_path(payload)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"success": False, "error": str(e), "detections": []}, ensure_ascii=False)

    max_pages = int(payload.get("max_pages", 1))
    detector = _get_yolo_detector()
    result = detector.detect(path, max_pages=max_pages)
    return json.dumps(result, ensure_ascii=False)
