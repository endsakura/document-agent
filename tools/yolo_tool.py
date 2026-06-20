"""YOLO 视觉识别 — 支持图片与 PDF（PDF 先转页图再检测）。"""
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

try:
    import fitz
except ImportError:
    fitz = None

from config import get_yolo_model_path
from tools.path_utils import resolve_file_path

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class YOLODetectionTool:
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25, pdf_zoom: float = 2.0):
        path = model_path or get_yolo_model_path()
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"YOLO 模型不存在: {path}\n"
                "请将 best.pt 放到 models/ 目录，或设置环境变量 YOLO_MODEL_PATH"
            )
        self.model = YOLO(path)
        self.conf_threshold = conf_threshold
        self.pdf_zoom = pdf_zoom

    def detect(self, source: str, max_pages: int = 1) -> Dict:
        """
        对图片或 PDF 做 YOLO 检测。
        PDF 会逐页渲染为图像后检测（默认首页，可通过 max_pages 扩展）。
        """
        try:
            path = str(resolve_file_path(source))
        except FileNotFoundError as e:
            return {"success": False, "error": str(e), "detections": []}

        ext = Path(path).suffix.lower()
        if ext in PDF_EXTENSIONS:
            return self._detect_pdf(path, max_pages=max_pages)
        if ext in IMAGE_EXTENSIONS:
            return self._detect_image(path, page=1)

        return {
            "success": False,
            "error": f"不支持的文件格式: {ext}",
            "detections": [],
        }

    def _detect_image(self, image_path: str, page: int = 1) -> Dict:
        try:
            detections = self._run_yolo(image_path)
            return {
                "success": True,
                "source": image_path,
                "page": page,
                "detections": detections,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "source": image_path,
                "page": page,
                "detections": [],
            }

    def _detect_pdf(self, pdf_path: str, max_pages: int = 1) -> Dict:
        if fitz is None:
            return {
                "success": False,
                "error": "缺少 PyMuPDF，无法对 PDF 做 YOLO。请安装：pip install pymupdf",
                "detections": [],
            }

        all_detections: List[Dict] = []
        pages_processed = 0

        try:
            doc = fitz.open(pdf_path)
            total = len(doc)
            to_process = min(max(1, max_pages), total)

            for page_index in range(to_process):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(self.pdf_zoom, self.pdf_zoom), alpha=False)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_path = tmp.name
                try:
                    pix.save(temp_path)
                    page_dets = self._run_yolo(temp_path)
                    pages_processed += 1
                    for det in page_dets:
                        det["page"] = page_index + 1
                        all_detections.append(det)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            doc.close()

            return {
                "success": True,
                "source": pdf_path,
                "file_type": "pdf",
                "total_pages": total,
                "pages_processed": pages_processed,
                "detections": all_detections,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "source": pdf_path,
                "file_type": "pdf",
                "detections": all_detections,
            }

    def _run_yolo(self, image_source: str) -> List[Dict]:
        results = self.model.predict(
            source=image_source,
            conf=self.conf_threshold,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                detections.append({
                    "class": self.model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),
                })
        return detections

    def should_use_ocr(self, detection_result: Dict) -> bool:
        if not detection_result.get("success") or not detection_result.get("detections"):
            return True
        return False


def create_yolo_tool(model_path: Optional[str] = None) -> YOLODetectionTool:
    return YOLODetectionTool(model_path)
