"""文档验证器 — 基于 PaddleOCR + 25 类规则（对接 tools.ocr）。"""
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

try:
    import fitz
except ImportError:
    fitz = None

from knowledge.rules_data import DOCUMENT_VALIDATORS
from knowledge.title_detection import (
    detect_document_type_by_title,
    infer_title_region_from_text,
    is_business_license_title_positioned,
    match_contract_title,
)
from tools import ocr as ocr_module


class DocumentValidator:
    """单文档类型验证器，OCR 底层使用 tools.ocr（PaddleOCR）。"""

    VALIDATORS = DOCUMENT_VALIDATORS

    def __init__(self, max_pages: int = 4):
        self.max_pages = max_pages
        self._title_region = ""
        self._ocr_items: List[dict] = []
        self._last_extracted_text = ""

    def get_supported_types(self) -> List[str]:
        return list(self.VALIDATORS.keys())

    def validate(self, file_path: str, doc_type: str) -> Dict:
        file_path = str(file_path)
        if not os.path.exists(file_path):
            return self._error_result(file_path, doc_type, f"文件不存在: {file_path}")

        if doc_type not in self.VALIDATORS:
            return self._error_result(file_path, doc_type, f"未知文档类型: {doc_type}")

        text = self._extract_text(file_path)
        if not text:
            return {
                "success": False,
                "file": os.path.basename(file_path),
                "doc_type": doc_type,
                "is_valid": False,
                "confidence": 0.0,
                "matched_keywords": [],
                "error": "无法提取文本内容",
            }

        result = self._score_document(text, doc_type)
        return {
            "success": result["is_valid"],
            "file": os.path.basename(file_path),
            "doc_type": doc_type,
            "is_valid": result["is_valid"],
            "confidence": result["confidence"],
            "matched_keywords": result["matched_keywords"],
            "detected_type": detect_document_type_by_title(text, self._title_region, self._ocr_items),
            "title_region": self._title_region,
            "text_preview": text[:2000],
            "error": result.get("reject_reason", "") if not result["is_valid"] else "",
        }

    def classify(self, file_path: str) -> Dict:
        """不指定类型，自动推断最可能的文档类型。"""
        text = self._extract_text(file_path)
        if not text:
            return {"success": False, "error": "无法提取文本"}

        title_type = detect_document_type_by_title(text, self._title_region, self._ocr_items)
        if title_type:
            validation = self._score_document(text, title_type)
            return {
                "success": True,
                "file": os.path.basename(file_path),
                "predicted_type": title_type,
                "method": "title_detection",
                "confidence": validation["confidence"],
                "is_valid": validation["is_valid"],
            }

        best_type, best_score, best_conf = "", 0, 0.0
        for doc_type in self.VALIDATORS:
            scored = self._score_document(text, doc_type)
            if scored["score"] > best_score:
                best_score = scored["score"]
                best_type = doc_type
                best_conf = scored["confidence"]

        return {
            "success": True,
            "file": os.path.basename(file_path),
            "predicted_type": best_type or "未知",
            "method": "keyword_scoring",
            "confidence": best_conf,
            "score": best_score,
        }

    def _extract_text(self, file_path: str) -> str:
        from tools.path_utils import resolve_file_path

        self._title_region = ""
        self._ocr_items = []
        try:
            file_path = str(resolve_file_path(file_path))
        except FileNotFoundError:
            return ""

        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            text = self._extract_pdf(file_path)
        elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"):
            text = self._extract_image(file_path)
        else:
            return ""

        self._last_extracted_text = text or ""
        return self._last_extracted_text

    def _extract_image(self, image_path: str) -> str:
        if hasattr(ocr_module, "extract_text_and_title_from_image"):
            full_text, title, items = ocr_module.extract_text_and_title_from_image(image_path)
            self._title_region = title or ""
            self._ocr_items = items or []
            return full_text or ""
        return ocr_module.extract_text_from_image(image_path) or ""

    def _extract_pdf(self, pdf_path: str) -> str:
        if fitz is None:
            pages = ocr_module.extract_text_from_pdf(pdf_path, max_pages=self.max_pages)
            text = "\n".join(pages)
            self._title_region = infer_title_region_from_text(text)
            return text

        doc = fitz.open(pdf_path)
        pages_to_read = min(self.max_pages, len(doc))
        texts = []
        for i in range(pages_to_read):
            page = doc[i]
            direct = (page.get_text() or "").strip()
            if len(direct) >= 50:
                texts.append(direct)
                if i == 0:
                    self._title_region = infer_title_region_from_text(direct)
            else:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    pix.save(tmp_path)
                    page_text, title, items = ocr_module.extract_text_and_title_from_image(tmp_path)
                    if page_text:
                        texts.append(page_text)
                    if i == 0:
                        self._title_region = title or ""
                        self._ocr_items = items or []
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        doc.close()
        return "\n".join(texts)

    def _score_document(self, text: str, doc_type: str) -> Dict:
        validator = self.VALIDATORS[doc_type]
        keywords = validator["keywords"]
        pass_score = validator["pass_score"]
        text_lower = text.lower()
        compact = re.sub(r"\s+", "", text)

        # 标题优先：标题识别类型与目标类型一致则直接通过
        detected = detect_document_type_by_title(text, self._title_region, self._ocr_items)
        if detected == doc_type:
            return {
                "is_valid": True,
                "confidence": 1.0,
                "matched_keywords": [f"标题识别:{detected}"],
                "score": 100,
            }

        # 合同标题强规则
        if doc_type == "合同":
            hit, kw = match_contract_title(self._title_region)
            if hit:
                return {"is_valid": True, "confidence": 1.0, "matched_keywords": [kw], "score": 100}

        # 营业执照位置校验
        if doc_type == "营业执照":
            if is_business_license_title_positioned(self._ocr_items, full_text=text):
                return {
                    "is_valid": True,
                    "confidence": 0.95,
                    "matched_keywords": ["营业执照(位置命中)"],
                    "score": 90,
                }

        # 排除词
        for ex in validator.get("exclude_keywords", []):
            if ex in text:
                return {
                    "is_valid": False,
                    "confidence": 0.0,
                    "matched_keywords": [f"排除:{ex}"],
                    "score": 0,
                    "reject_reason": f"命中排除词: {ex}",
                }

        total_score = 0
        matched = []
        for keyword, weight in keywords.items():
            if keyword.lower() in text_lower or keyword in compact:
                total_score += weight
                matched.append(f"{keyword}(+{weight})")

        is_valid = total_score >= pass_score
        confidence = min(total_score / pass_score, 1.0) if pass_score else 0.0
        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "matched_keywords": matched,
            "score": total_score,
            "reject_reason": "" if is_valid else "关键词得分不足",
        }

    @staticmethod
    def _error_result(file_path: str, doc_type: str, error: str) -> Dict:
        return {
            "success": False,
            "file": os.path.basename(file_path),
            "doc_type": doc_type,
            "is_valid": False,
            "confidence": 0.0,
            "matched_keywords": [],
            "error": error,
        }
