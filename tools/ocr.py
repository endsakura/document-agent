import os
import json
import tempfile
import random
from collections import Counter
from functools import lru_cache

from paddleocr import PaddleOCR

# OCR 结果缓存条目数（按 路径+mtime+size 区分）
OCR_CACHE_MAXSIZE = 256

# ========= 全局OCR（关键优化：只初始化一次） =========
ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
    text_det_box_thresh=0.3
)

# ========= 标签配置 =========
LABEL_MAP = {
    "contract": ("0", "合同"),
    "report": ("1", "财报"),
    "bid": ("2", "中标通知书"),
    "company": ("3", "公司章程"),
    "cr": ("4", "登记证明")
}

DATA_DIR = r"D:\image\新建文件夹"
OUTPUT_FILE = "train.txt"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".pdf")


def normalize_text(text):
    return " ".join(str(text).replace("\n", " ").replace("\r", " ").split()).strip()


def _bbox_to_rect(poly):
    """将多边形 bbox 转为 (xmin, ymin, xmax, ymax)。"""
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _file_cache_key(image_path):
    """生成可哈希的缓存键；文件不存在或临时文件删除后返回 None。"""
    try:
        abspath = os.path.abspath(image_path)
        stat = os.stat(abspath)
        return abspath, stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _items_to_tuple(items):
    return tuple(
        (item["text"], item["xmin"], item["ymin"], item["xmax"], item["ymax"])
        for item in items
    )


def _tuple_to_items(items_tuple):
    return [
        {"text": t, "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1}
        for t, x0, y0, x1, y1 in items_tuple
    ]


@lru_cache(maxsize=OCR_CACHE_MAXSIZE)
def _collect_ocr_items_cached(abspath: str, mtime_ns: int, size: int):
    """带 lru_cache 的 OCR 采集（键含 mtime，文件更新后自动失效）。"""
    return _items_to_tuple(_collect_ocr_items_uncached(abspath))


def _collect_ocr_items_uncached(image_path):
    """
    从 PaddleOCR 结果中收集带坐标的文本块（无缓存）。
    返回: [{"text", "xmin", "ymin", "xmax", "ymax"}, ...]
    """
    items = []

    # 新版 PaddleOCR predict 接口
    try:
        results = ocr.predict(image_path)
        for res in results or []:
            rec_texts = res.get("rec_texts") or []
            polys = (
                res.get("dt_polys")
                or res.get("rec_polys")
                or res.get("rec_boxes")
                or []
            )
            for idx, text in enumerate(rec_texts):
                text = str(text).strip()
                if not text:
                    continue
                if idx < len(polys):
                    poly = polys[idx]
                    if hasattr(poly, "tolist"):
                        poly = poly.tolist()
                    if len(poly) >= 4:
                        if len(poly) == 4 and not isinstance(poly[0], (list, tuple)):
                            xmin, ymin, xmax, ymax = map(float, poly)
                        else:
                            xmin, ymin, xmax, ymax = _bbox_to_rect(poly)
                        items.append({
                            "text": text,
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                        })
    except Exception:
        pass

    # 兼容旧版 ocr 接口
    if not items:
        try:
            results = ocr.ocr(image_path, cls=True)
            if results and results[0]:
                for line in results[0]:
                    if len(line) < 2 or not line[1]:
                        continue
                    text = str(line[1][0]).strip()
                    if not text:
                        continue
                    xmin, ymin, xmax, ymax = _bbox_to_rect(line[0])
                    items.append({
                        "text": text,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    })
        except Exception:
            pass

    return items


def _collect_ocr_items(image_path):
    """收集 OCR 文本块；相同文件（路径+mtime+size）命中 lru_cache。"""
    cache_key = _file_cache_key(image_path)
    if cache_key is None:
        return _collect_ocr_items_uncached(image_path)
    abspath, mtime_ns, size = cache_key
    cached = _collect_ocr_items_cached(abspath, mtime_ns, size)
    return _tuple_to_items(cached)


def clear_ocr_cache():
    """清空 OCR 缓存（测试或批量更新图片后可选调用）。"""
    _collect_ocr_items_cached.cache_clear()


def _title_items_from_ocr_items(items, top_ratio=0.2):
    """按 bbox 取页面上方 top_ratio 区域内的文本块（自上而下排序）。"""
    if not items:
        return []
    page_height = max(item["ymax"] for item in items)
    if page_height <= 0:
        return []
    threshold = page_height * top_ratio
    title_items = [item for item in items if item["ymin"] < threshold]
    title_items.sort(key=lambda x: (x["ymin"], x["xmin"]))
    return title_items


def extract_title_region_from_image(image_path, top_ratio=0.2):
    """基于 bbox 提取页面上方区域作为标题（默认上方 20%）。"""
    items = _collect_ocr_items(image_path)
    title_items = _title_items_from_ocr_items(items, top_ratio)
    if title_items:
        return normalize_text(" ".join(item["text"] for item in title_items))
    # 无 bbox 时回退到全文前段
    full_text = extract_text_from_image(image_path)
    return full_text[: max(50, int(len(full_text) * top_ratio))] if full_text else ""


def collect_ocr_items(image_path):
    """收集带 bbox 的 OCR 文本块（供位置校验使用）。"""
    return _collect_ocr_items(image_path)


def _item_center(item):
    return (item["xmin"] + item["xmax"]) / 2, (item["ymin"] + item["ymax"]) / 2


def _in_center_upper_band(cx, cy, page_width, page_height, x_min_ratio, x_max_ratio, y_min_ratio, y_max_ratio):
    return (
        page_width * x_min_ratio <= cx <= page_width * x_max_ratio
        and page_height * y_min_ratio <= cy <= page_height * y_max_ratio
    )


def is_keyword_in_center_upper_region(
    items,
    keyword,
    x_min_ratio=0.12,
    x_max_ratio=0.88,
    y_min_ratio=0.02,
    y_max_ratio=0.50,
):
    """判断 keyword 是否出现在页面「水平居中、偏上」区域。"""
    if not items or not keyword:
        return False

    page_width = max(item["xmax"] for item in items)
    page_height = max(item["ymax"] for item in items)
    if page_width <= 0 or page_height <= 0:
        return False

    for item in items:
        if keyword not in item["text"].replace(" ", ""):
            continue
        cx, cy = _item_center(item)
        if _in_center_upper_band(
            cx, cy, page_width, page_height,
            x_min_ratio, x_max_ratio, y_min_ratio, y_max_ratio,
        ):
            return True
    return False


def _license_title_in_top_center_by_rank(items, top_fraction=0.35, x_margin_ratio=0.18):
    """
    按纵向排序取最上方若干文本块，再判断水平居中区域是否含「营业」「执照」。
    不依赖绝对像素比例，避免 bbox 坐标系异常导致误判。
    """
    if not items:
        return False
    sorted_items = sorted(items, key=lambda x: (x["ymin"], x["xmin"]))
    top_count = max(1, int(len(sorted_items) * top_fraction))
    top_items = sorted_items[:top_count]
    page_width = max(item["xmax"] for item in items)
    if page_width <= 0:
        return False
    x_lo = page_width * x_margin_ratio
    x_hi = page_width * (1 - x_margin_ratio)
    center_top = []
    for item in top_items:
        cx, _ = _item_center(item)
        if x_lo <= cx <= x_hi:
            center_top.append(item)
    if not center_top:
        center_top = top_items
    texts = "".join(i["text"].replace(" ", "") for i in center_top)
    if "营业执照" in texts:
        return True
    has_yingye = any("营业" in i["text"] for i in center_top)
    has_zhizhao = any("执照" in i["text"] for i in center_top)
    return has_yingye and has_zhizhao


def is_business_license_title_positioned(
    items,
    full_text=None,
    x_min_ratio=0.12,
    x_max_ratio=0.88,
    y_min_ratio=0.02,
    y_max_ratio=0.50,
):
    """
    判断「营业执照」是否在证照正中偏上。
    兼容：整词命中、OCR 拆成「营业」+「执照」、盖章/去污后 bbox 偏移。
    """
    if items and _license_title_in_top_center_by_rank(items):
        return True

    if items:
        page_width = max(item["xmax"] for item in items)
        page_height = max(item["ymax"] for item in items)
        if page_width > 0 and page_height > 0:
            # 略放宽的垂直范围，用于拼接「营业」「执照」分块
            y_hi_split = min(0.55, y_max_ratio + 0.10)
            band_items = []
            for item in items:
                cx, cy = _item_center(item)
                if _in_center_upper_band(
                    cx, cy, page_width, page_height,
                    x_min_ratio, x_max_ratio, y_min_ratio, y_hi_split,
                ):
                    band_items.append(item)

            for item in band_items:
                t = item["text"].replace(" ", "")
                if "营业执照" in t:
                    return True

            has_yingye = any("营业" in i["text"].replace(" ", "") for i in band_items)
            has_zhizhao = any("执照" in i["text"].replace(" ", "") for i in band_items)
            if has_yingye and has_zhizhao:
                return True

            if is_keyword_in_center_upper_region(
                items, "营业执照",
                x_min_ratio, x_max_ratio, y_min_ratio, y_max_ratio,
            ):
                return True

            # 上方区域出现「营业」或「执照」之一（OCR 漏字时）
            upper_partial = any(
                "营业" in i["text"] or "执照" in i["text"]
                for i in band_items
            )
            if upper_partial and full_text:
                compact = str(full_text).replace(" ", "").replace("\n", "")
                if "营业执照" in compact or ("统一社会信用代码" in compact and "法定代表人" in compact):
                    return True

    if not full_text:
        return False

    compact = str(full_text).replace(" ", "").replace("\n", "")
    has_title = (
        "营业执照" in compact
        or ("营业" in compact and "执照" in compact)
    )
    if not has_title:
        return False

    # 无 bbox 或 bbox 全失败：正文强特征兜底（扫描 PDF 直接抽字、测试脚本仅传文本）
    strong_markers = [
        "统一社会信用代码", "登记机关", "法定代表人",
        "经营范围", "注册资本", "营业期限", "成立日期",
    ]
    marker_count = sum(1 for m in strong_markers if m in compact)
    return marker_count >= 2


def extract_text_and_title_from_image(image_path, top_ratio=0.2):
    """
    同时返回全文、标题区域文本、OCR 文本块（含 bbox）。
    全文按阅读顺序（上到下、左到右）拼接，标题取页面上方 top_ratio 区域。
    """
    items = _collect_ocr_items(image_path)
    if items:
        sorted_items = sorted(items, key=lambda x: (x["ymin"], x["xmin"]))
        full_text = normalize_text(" ".join(item["text"] for item in sorted_items))
        title_items = _title_items_from_ocr_items(items, top_ratio)
        title_text = normalize_text(" ".join(item["text"] for item in title_items))
        return full_text, title_text, items

    full_text = normalize_text("")
    try:
        results = ocr.predict(image_path)
        for res in results or []:
            rec_texts = res.get("rec_texts") or []
            if rec_texts:
                full_text = normalize_text(" ".join(str(t) for t in rec_texts))
                break
    except Exception:
        pass
    if not full_text:
        try:
            results = ocr.ocr(image_path, cls=True)
            if results and results[0]:
                texts = [line[1][0] for line in results[0] if len(line) > 1 and line[1]]
                full_text = normalize_text(" ".join(texts))
        except Exception:
            pass

    fallback_title = full_text[: max(50, int(len(full_text) * top_ratio))] if full_text else ""
    return full_text, fallback_title, []


def extract_text_from_image(image_path):
    """对图片做OCR，只返回纯文本（按 bbox 阅读顺序拼接）。"""
    try:
        full_text, _, _ = extract_text_and_title_from_image(image_path)
        return full_text
    except Exception as e:
        print(f"处理图片失败: {image_path} -> {e}")
        return ""


def extract_text_from_pdf(pdf_path, max_pages=None, min_chars_per_page=50, zoom_levels=(3, 4)):
    """
    将 PDF 逐页转图片后做 OCR，返回每一页的文本列表。
    扫描件 PDF：逐页渲染为高清图再 PaddleOCR。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("缺少 PyMuPDF（fitz），请先安装：pip install pymupdf")
        return []

    from tools.path_utils import resolve_file_path

    try:
        pdf_file = resolve_file_path(pdf_path)
    except FileNotFoundError as e:
        print(f"PDF 路径无效: {e}")
        return []

    page_texts = []

    try:
        doc = fitz.open(str(pdf_file))
        total_pages = len(doc)
        if total_pages == 0:
            print(f"PDF 无页面: {pdf_file}")
            doc.close()
            return []

        pages_to_process = total_pages if max_pages is None else min(max_pages, total_pages)
        print(f"处理 PDF: {pdf_file}，共 {total_pages} 页，处理前 {pages_to_process} 页")

        for page_index in range(pages_to_process):
            page = doc.load_page(page_index)
            direct_text = normalize_text(page.get_text() or "")

            if len(direct_text) >= min_chars_per_page:
                page_text = direct_text
                print(
                    f"    第 {page_index + 1}/{total_pages} 页可编辑 PDF，"
                    f"直接提取 {len(page_text)} 字"
                )
            else:
                page_text = ""
                for zoom in zoom_levels:
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_image_path = temp_file.name
                    try:
                        pix.save(temp_image_path)
                        page_text = normalize_text(extract_text_from_image(temp_image_path))
                        if page_text:
                            print(
                                f"    第 {page_index + 1}/{total_pages} 页 OCR 成功 "
                                f"({zoom}x)，{len(page_text)} 字"
                            )
                            break
                        print(
                            f"    第 {page_index + 1}/{total_pages} 页 "
                            f"{zoom}x OCR 无结果，尝试更高分辨率..."
                        )
                    finally:
                        if os.path.exists(temp_image_path):
                            os.remove(temp_image_path)

            if page_text:
                page_texts.append(page_text)
            else:
                print(f"    第 {page_index + 1}/{total_pages} 页无有效文本")

        doc.close()
        return page_texts
    except Exception as e:
        print(f"处理PDF失败: {pdf_file} -> {e}")
        import traceback
        traceback.print_exc()
        return []


def process_category(category_name, label_id, label_des):
    """处理单个分类目录。"""
    category_dir = os.path.join(DATA_DIR, category_name)

    if not os.path.exists(category_dir):
        print(f"目录不存在: {category_dir}")
        return []

    results = []
    files = [
        f for f in os.listdir(category_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS + (".pdf",))
    ]

    print(f"处理 {label_des} ({category_name}): 找到 {len(files)} 个文件")

    for idx, file_name in enumerate(files, 1):
        file_path = os.path.join(category_dir, file_name)
        lower_name = file_name.lower()

        print(f"  [{idx}/{len(files)}] 开始处理: {file_name}")

        if lower_name.endswith(".pdf"):
            page_sentences = extract_text_from_pdf(file_path)

            if page_sentences:
                for page_idx, sentence in enumerate(page_sentences, 1):
                    sentence = normalize_text(sentence)
                    if not sentence:
                        continue

                    data = {
                        "label": label_id,
                        "label_des": label_des,
                        "sentence": sentence
                    }
                    results.append(data)
                    print(f"  [{idx}/{len(files)}] 第{page_idx}页已完成，文本长度: {len(sentence)}")
            else:
                print(f"  [{idx}/{len(files)}] 跳过，未识别到有效文本")
        else:
            sentence = extract_text_from_image(file_path)
            sentence = normalize_text(sentence)

            if sentence:
                data = {
                    "label": label_id,
                    "label_des": label_des,
                    "sentence": sentence
                }
                results.append(data)
                print(f"  [{idx}/{len(files)}] 已完成，文本长度: {len(sentence)}")
            else:
                print(f"  [{idx}/{len(files)}] 跳过，未识别到有效文本")

    return results


def balance_data(all_data, strategy="mixed"):
    """
    平衡各类数据数量。

    Args:
        all_data: 所有数据列表
        strategy: 平衡策略
            - "mixed": 混合方案（推荐）- 先过采样少数类，再欠采样多数类
            - "oversample": 过采样 - 复制少数类到最多类的数量
            - "undersample": 欠采样 - 删除多数类到最少类的数量

    Returns:
        平衡后的数据列表
    """
    # 按标签分组
    label_groups = {}
    for item in all_data:
        label = item["label"]
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)

    # 统计各类数据
    label_counts = {label: len(items) for label, items in label_groups.items()}
    min_count = min(label_counts.values())
    max_count = max(label_counts.values())
    target_count = int((min_count + max_count) / 2)  # 目标数量为最小和最大的平均值

    print(f"\n========== 数据平衡统计 ==========")
    print(f"平衡前各类数据数量:")
    for label, count in sorted(label_counts.items()):
        label_des = next(v[1] for k, v in LABEL_MAP.items() if v[0] == label)
        print(f"  标签 {label} ({label_des}): {count} 条")
    print(f"最少类: {min_count} 条，最多类: {max_count} 条")
    print(f"目标数量: {target_count} 条")
    print(f"采用策略: {strategy}")

    balanced_data = []

    if strategy == "mixed":
        # 混合方案：先过采样少数类，再欠采样多数类
        for label, items in label_groups.items():
            current_count = len(items)

            if current_count < target_count:
                # 过采样：复制数据
                shortage = target_count - current_count
                sampled = random.choices(items, k=shortage)
                balanced_data.extend(items + sampled)
                print(f"  标签 {label}: 过采样 {shortage} 条 ({current_count} → {target_count})")
            elif current_count > target_count:
                # 欠采样：随机删除
                sampled = random.sample(items, target_count)
                balanced_data.extend(sampled)
                print(f"  标签 {label}: 欠采样 {current_count - target_count} 条 ({current_count} → {target_count})")
            else:
                # 数量相同
                balanced_data.extend(items)
                print(f"  标签 {label}: 保持不变 ({current_count} 条)")

    elif strategy == "oversample":
        # 过采样方案：所有类都采样到最多类的数量
        for label, items in label_groups.items():
            current_count = len(items)
            if current_count < max_count:
                shortage = max_count - current_count
                sampled = random.choices(items, k=shortage)
                balanced_data.extend(items + sampled)
                print(f"  标签 {label}: 过采样 {shortage} 条 ({current_count} → {max_count})")
            else:
                balanced_data.extend(items)
                print(f"  标签 {label}: 保持不变 ({current_count} 条)")

    elif strategy == "undersample":
        # 欠采样方案：所有类都采样到最少类的数量
        for label, items in label_groups.items():
            current_count = len(items)
            if current_count > min_count:
                sampled = random.sample(items, min_count)
                balanced_data.extend(sampled)
                print(f"  标签 {label}: 欠采样 {current_count - min_count} 条 ({current_count} → {min_count})")
            else:
                balanced_data.extend(items)
                print(f"  标签 {label}: 保持不变 ({current_count} 条)")

    # 打乱顺序
    random.shuffle(balanced_data)

    print(f"\n平衡后各类数据数量:")
    balanced_counts = Counter(item["label"] for item in balanced_data)
    for label in sorted(balanced_counts.keys()):
        label_des = next(v[1] for k, v in LABEL_MAP.items() if v[0] == label)
        print(f"  标签 {label} ({label_des}): {balanced_counts[label]} 条")
    print(f"总计: {len(balanced_data)} 条")
    print(f"==================================\n")

    return balanced_data


def main():
    all_data = []

    for category_name, (label_id, label_des) in LABEL_MAP.items():
        category_data = process_category(category_name, label_id, label_des)
        all_data.extend(category_data)

    print(f"\n原始数据统计:")
    original_counts = Counter(item["label"] for item in all_data)
    for label in sorted(original_counts.keys()):
        label_des = next(v[1] for k, v in LABEL_MAP.items() if v[0] == label)
        print(f"  标签 {label} ({label_des}): {original_counts[label]} 条")
    print(f"总计: {len(all_data)} 条\n")

    # 平衡数据（使用混合方案）
    balanced_data = balance_data(all_data, strategy="mixed")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in balanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✓ 完成！共生成 {len(balanced_data)} 条训练数据")
    print(f"✓ 输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
