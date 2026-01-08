# ==============================================================================
#  HYBRID OMR (ROBUST) - Auto Train from Answer Key + Student Grading
#  - Fixes: ID vs Questions swap, missing C/D columns, X-cancel logic
#  - Supports: choices = 2/4/5, questions = variable, ID = 4 digits x 10 rows
#  - UI: shows extracted Answer Key + counts + overlays before grading
# ==============================================================================

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# pdf support
try:
    from pdf2image import convert_from_bytes
    _HAS_PDF2IMAGE = True
except Exception:
    _HAS_PDF2IMAGE = False

try:
    import fitz  # PyMuPDF fallback
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False


# ==============================================================================
# Data models
# ==============================================================================

@dataclass
class GridModel:
    # canonical page shape (after resize)
    width: int
    height: int

    # ID grid fixed
    id_digits: int = 4
    id_rows: int = 10
    id_col_x: Optional[List[float]] = None
    id_row_y: Optional[List[float]] = None
    id_r: float = 18.0

    # Questions grid variable
    q_cols: int = 4  # 2/4/5
    q_rows: int = 10
    q_col_x: Optional[List[float]] = None
    q_row_y: Optional[List[float]] = None
    q_r: float = 18.0

    # For mapping expected centers to detected circles
    expected_id_centers: Optional[List[Tuple[float, float]]] = None
    expected_q_centers: Optional[List[Tuple[float, float]]] = None


@dataclass
class DetectParams:
    # bubble fill
    blank_fill_threshold: float = 0.16
    double_ratio: float = 1.35

    # X cancel detection
    x_hough_min_len_factor: float = 0.90   # min line length ~ r * factor
    x_threshold: float = 1.00              # x_score threshold

    # circle detection (contours)
    min_area: int = 200
    max_area: int = 12000
    min_circularity: float = 0.45


# ==============================================================================
# Image loading
# ==============================================================================

def load_first_page(file_bytes: bytes, filename: str, dpi: int = 300) -> Image.Image:
    name = (filename or "").lower()

    if name.endswith(".pdf"):
        if _HAS_PDF2IMAGE:
            pages = convert_from_bytes(file_bytes, dpi=dpi)
            if not pages:
                raise ValueError("PDF فارغ أو لا يمكن قراءته.")
            return pages[0].convert("RGB")
        if _HAS_PYMUPDF:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if doc.page_count == 0:
                raise ValueError("PDF فارغ أو لا يمكن قراءته.")
            page = doc.load_page(0)
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img
        raise ValueError("لا يوجد pdf2image ولا PyMuPDF. ثبّت أحدهما لقراءة PDF.")
    else:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def resize_to(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)


# ==============================================================================
# Circle detection (robust): Hough + Contours + fallback
# ==============================================================================

def enhance_for_circles(gray: np.ndarray) -> np.ndarray:
    # CLAHE makes thin circle borders pop
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return g


def detect_circles_hough(img_bgr: np.ndarray,
                         dp: float = 1.2,
                         min_dist: int = 24,
                         param1: int = 140,
                         param2: int = 22,
                         min_r: int = 9,
                         max_r: int = 52) -> List[Tuple[float, float, float]]:
    g0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = enhance_for_circles(g0)
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_r, maxRadius=max_r
    )
    out: List[Tuple[float, float, float]] = []
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            out.append((float(x), float(y), float(r)))
    return out


def detect_circles_contours(img_bgr: np.ndarray,
                            min_area: int = 200,
                            max_area: int = 12000,
                            min_circularity: float = 0.45) -> List[Tuple[float, float, float]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = enhance_for_circles(gray)

    edges = cv2.Canny(g, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles: List[Tuple[float, float, float]] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(c, True)
        if peri <= 1e-6:
            continue

        circ = 4 * np.pi * area / (peri * peri)
        if circ < min_circularity:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        if r < 7 or r > 70:
            continue

        circles.append((float(x), float(y), float(r)))

    return circles


def merge_circles(c1: List[Tuple[float, float, float]],
                  c2: List[Tuple[float, float, float]],
                  tol: float = 10.0) -> List[Tuple[float, float, float]]:
    # merge near duplicates
    out = list(c1)
    for (x2, y2, r2) in c2:
        ok = True
        for (x1, y1, r1) in out:
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 < tol ** 2:
                ok = False
                break
        if ok:
            out.append((x2, y2, r2))
    return out


def detect_circles_auto(img_bgr: np.ndarray, dp: DetectParams) -> List[Tuple[float, float, float]]:
    # Try Hough, then contour, and merge
    h1 = detect_circles_hough(img_bgr, param2=22, min_dist=24, min_r=9, max_r=52)
    h2 = detect_circles_hough(img_bgr, param2=19, min_dist=22, min_r=8, max_r=55)
    c3 = detect_circles_contours(img_bgr, dp.min_area, dp.max_area, dp.min_circularity)
    merged = merge_circles(merge_circles(h1, h2, tol=9.0), c3, tol=9.0)
    return merged


# ==============================================================================
# 1D clustering helpers
# ==============================================================================

def kmeans_1d(vals: np.ndarray, k: int, iters: int = 30) -> np.ndarray:
    vals = vals.astype(np.float32).reshape(-1, 1)
    if len(vals) < k:
        raise ValueError("Not enough points for kmeans.")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1e-3)
    _compactness, labels, centers = cv2.kmeans(vals, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = centers.flatten()
    centers.sort()
    return centers


def group_by_proximity(sorted_vals: np.ndarray, tol: float) -> np.ndarray:
    if len(sorted_vals) == 0:
        return np.array([])
    groups = [[float(sorted_vals[0])]]
    for v in sorted_vals[1:]:
        if abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(float(v))
        else:
            groups.append([float(v)])
    centers = np.array([np.mean(g) for g in groups], dtype=np.float32)
    return centers


def median_nn_dist(vals: np.ndarray) -> float:
    if len(vals) < 2:
        return 0.0
    v = np.sort(vals)
    d = np.diff(v)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 0.0


# ==============================================================================
# Grid finding: ID (4x10) + Questions (2/4/5 x variable rows)
# ==============================================================================

def find_id_grid(circles: List[Tuple[float, float, float]], w: int, h: int,
                 id_digits: int = 4, id_rows: int = 10) -> Tuple[List[float], List[float], float, Dict]:
    """
    Robustly locate ID grid in TOP-RIGHT area (based on your sheet layout).
    """
    # candidate ROI: top-right
    roi = [(x, y, r) for (x, y, r) in circles if (x > 0.55 * w and y < 0.60 * h)]
    dbg = {"roi_count": len(roi)}

    # fallback: if too few, also try top-left (in case of mirrored scan)
    if len(roi) < 20:
        roi = [(x, y, r) for (x, y, r) in circles if (x < 0.45 * w and y < 0.60 * h)]
        dbg["roi_fallback_left"] = True
        dbg["roi_count"] = len(roi)

    if len(roi) < 20:
        raise ValueError("لم أستطع إيجاد دوائر كافية لمنطقة كود الطالب. تأكد أن الصفحة كاملة و DPI عالي.")

    xs = np.array([c[0] for c in roi], dtype=np.float32)
    ys = np.array([c[1] for c in roi], dtype=np.float32)
    rs = np.array([c[2] for c in roi], dtype=np.float32)
    r_med = float(np.median(rs))

    # columns (4)
    col_x = kmeans_1d(xs, id_digits).tolist()
    # rows (10)
    row_y = kmeans_1d(ys, id_rows).tolist()

    # validate grid coverage
    grid_hits = 0
    for cy in row_y:
        for cx in col_x:
            best = None
            best_d = 1e18
            for (x, y, r) in roi:
                d = (x - cx) ** 2 + (y - cy) ** 2
                if d < best_d:
                    best_d = d
                    best = (x, y, r)
            if best is not None and math.sqrt(best_d) <= 2.2 * r_med:
                grid_hits += 1

    dbg["grid_hits"] = grid_hits
    dbg["expected"] = id_digits * id_rows
    dbg["r_med"] = r_med

    if grid_hits < int(0.70 * id_digits * id_rows):
        raise ValueError("فشل تحديد شبكة كود الطالب 4×10 (تداخل/قص/وضوح). ارفع DPI أو تأكد أن الورقة كاملة.")

    return col_x, row_y, r_med, dbg


def score_cols(xs: np.ndarray, k: int) -> float:
    """
    prefer wider spread + near-uniform spacing
    """
    centers = kmeans_1d(xs, k)
    if len(centers) < 2:
        return -1e9
    dif = np.diff(centers)
    if np.any(dif <= 1e-6):
        return -1e9
    spread = float(centers[-1] - centers[0])
    uniformity = float(np.std(dif) / (np.mean(dif) + 1e-9))
    # higher better
    return spread - 500.0 * uniformity


def find_question_grid(circles: List[Tuple[float, float, float]], w: int, h: int,
                       choices_candidates=(2, 4, 5)) -> Tuple[List[float], List[float], float, Dict]:
    """
    Locate question bubbles in lower/left area, detect q_cols in {2,4,5} and q_rows variable.
    """
    # remove ID area (top-right)
    filtered = [(x, y, r) for (x, y, r) in circles if not (x > 0.55 * w and y < 0.60 * h)]
    # also remove header noise (very top)
    filtered = [(x, y, r) for (x, y, r) in filtered if y > 0.25 * h]
    dbg = {"filtered_count": len(filtered)}

    if len(filtered) < 12:
        raise ValueError("دوائر الأسئلة قليلة جداً. تأكد من DPI ووضوح الورقة (والصفحة كاملة).")

    xs = np.array([c[0] for c in filtered], dtype=np.float32)
    ys = np.array([c[1] for c in filtered], dtype=np.float32)
    rs = np.array([c[2] for c in filtered], dtype=np.float32)
    r_med = float(np.median(rs))

    # choose q_cols by scoring
    best_k = None
    best_score = -1e18
    for k in choices_candidates:
        if len(xs) < k:
            continue
        try:
            s = score_cols(xs, k)
            if s > best_score:
                best_score = s
                best_k = k
        except Exception:
            continue

    if best_k is None:
        raise ValueError("لم أستطع تحديد عدد الخيارات (2/4/5).")

    q_col_x = kmeans_1d(xs, best_k).tolist()

    # rows: group y by proximity (no fixed count)
    y_sorted = np.sort(ys)
    dy = median_nn_dist(y_sorted)
    if dy <= 0:
        raise ValueError("تعذر تقدير تباعد صفوف الأسئلة.")
    tol = max(10.0, 0.65 * dy)
    q_row_y = group_by_proximity(y_sorted, tol=tol).tolist()

    # Remove very short groups by checking row support
    kept_rows: List[float] = []
    for ry in q_row_y:
        # count circles near this row
        count = 0
        for (x, y, r) in filtered:
            if abs(y - ry) <= 0.9 * dy:
                count += 1
        # row should roughly have >= k circles
        if count >= max(2, best_k - 1):
            kept_rows.append(float(ry))

    kept_rows.sort()
    q_row_y = kept_rows

    # Sometimes first row is noisy (letters A/B/C/D row) — drop if it has too few circles
    if len(q_row_y) > 2:
        # heuristic: compare gaps, keep consistent band
        gaps = np.diff(np.array(q_row_y))
        medg = float(np.median(gaps)) if len(gaps) else 0.0
        if medg > 0:
            good = [q_row_y[0]]
            for i in range(1, len(q_row_y)):
                if abs((q_row_y[i] - q_row_y[i-1]) - medg) <= 0.55 * medg:
                    good.append(q_row_y[i])
            if len(good) >= 2:
                q_row_y = good

    dbg.update({
        "q_cols": best_k,
        "q_rows": len(q_row_y),
        "r_med": r_med,
        "dy": dy,
        "tol": tol
    })

    if len(q_row_y) < 3:
        raise ValueError("لم أستطع تحديد صفوف الأسئلة بشكل موثوق. ربما الصفحة مقصوصة أو الإضاءة سيئة.")

    return q_col_x, q_row_y, r_med, dbg


# ==============================================================================
# Bubble scoring: fill + X detection
# ==============================================================================

def circle_masks(r: int, inner_scale: float = 0.72) -> Tuple[np.ndarray, np.ndarray]:
    size = int(2 * r + 5)
    cx = cy = size // 2
    y, x = np.ogrid[:size, :size]
    dist2 = (x - cx) ** 2 + (y - cy) ** 2
    outer = dist2 <= (r * 0.98) ** 2
    inner = dist2 <= (r * inner_scale) ** 2
    ring = outer & (~inner)   # circle border area
    return inner.astype(np.uint8), ring.astype(np.uint8)


def extract_roi(gray: np.ndarray, x: float, y: float, r: float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = gray.shape[:2]
    rr = int(max(8, r))
    x1 = max(0, int(x - rr - 3))
    y1 = max(0, int(y - rr - 3))
    x2 = min(w, int(x + rr + 3))
    y2 = min(h, int(y + rr + 3))
    roi = gray[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


def binarize_ink(gray_roi: np.ndarray) -> np.ndarray:
    # ink = 1
    g = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 6)
    return (th > 0).astype(np.uint8)


def fill_ratio_and_xscore(gray_page: np.ndarray, x: float, y: float, r: float,
                          dp: DetectParams) -> Tuple[float, float]:
    roi, _ = extract_roi(gray_page, x, y, r)
    if roi.size == 0:
        return 0.0, 0.0

    ink = binarize_ink(roi)

    rr = int(max(8, r))
    inner_m, ring_m = circle_masks(rr, inner_scale=0.72)

    # align mask to ROI size
    mh, mw = inner_m.shape[:2]
    ih, iw = ink.shape[:2]
    # center crop/pad to match
    if ih != mh or iw != mw:
        ink_resized = cv2.resize(ink.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST)
    else:
        ink_resized = ink

    inner_pixels = ink_resized[inner_m.astype(bool)]
    fill = float(np.mean(inner_pixels)) if inner_pixels.size else 0.0

    # X detection: look for diagonal lines inside inner+ring area (ignore outside)
    x_score = 0.0
    region = (inner_m | ring_m).astype(np.uint8)
    region_ink = (ink_resized & region)

    edges = cv2.Canny((region_ink * 255).astype(np.uint8), 40, 130)
    min_len = int(max(10, rr * dp.x_hough_min_len_factor))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                            minLineLength=min_len, maxLineGap=6)
    if lines is not None:
        diag_count = 0
        diag_len_sum = 0.0
        for (x1, y1, x2, y2) in lines[:, 0, :]:
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < min_len:
                continue
            ang = abs(math.degrees(math.atan2(dy, dx)))
            # diagonal-ish
            if (25 <= ang <= 65) or (115 <= ang <= 155):
                diag_count += 1
                diag_len_sum += length
        x_score = float(diag_count) + float(diag_len_sum / (rr + 1e-9)) * 0.15

    return fill, x_score


def pick_answer_for_row(fills: List[float], xscores: List[float], choices: List[str],
                        dp: DetectParams) -> Dict:
    # cancelled mask
    cancelled = [xscores[i] >= dp.x_threshold for i in range(len(choices))]

    # candidates: non-cancelled first
    cand = [i for i in range(len(choices)) if not cancelled[i]]
    if not cand:
        # all cancelled -> treat as blank
        return {"answer": "?", "status": "BLANK", "debug": {"cancelled": cancelled, "fills": fills, "xs": xscores}}

    # choose max fill among non-cancelled
    cand_sorted = sorted(cand, key=lambda i: fills[i], reverse=True)
    top = cand_sorted[0]
    top_fill = fills[top]
    second_fill = fills[cand_sorted[1]] if len(cand_sorted) > 1 else 0.0

    if top_fill < dp.blank_fill_threshold:
        return {"answer": "?", "status": "BLANK", "debug": {"cancelled": cancelled, "fills": fills, "xs": xscores}}

    # double mark (only among non-cancelled)
    if second_fill >= dp.blank_fill_threshold and (top_fill / (second_fill + 1e-9)) < dp.double_ratio:
        return {"answer": "!", "status": "DOUBLE", "debug": {"cancelled": cancelled, "fills": fills, "xs": xscores}}

    return {"answer": choices[top], "status": "OK", "debug": {"cancelled": cancelled, "fills": fills, "xs": xscores}}


# ==============================================================================
# Grid -> centers
# ==============================================================================

def build_centers(col_x: List[float], row_y: List[float]) -> List[Tuple[float, float]]:
    centers = []
    for ry in row_y:
        for cx in col_x:
            centers.append((float(cx), float(ry)))
    return centers


def nearest_circle_center(expected: Tuple[float, float],
                          circles: List[Tuple[float, float, float]],
                          max_dist: float) -> Tuple[float, float, float]:
    ex, ey = expected
    best = None
    best_d = 1e18
    for (x, y, r) in circles:
        d = (x - ex) ** 2 + (y - ey) ** 2
        if d < best_d:
            best_d = d
            best = (x, y, r)
    if best is not None and math.sqrt(best_d) <= max_dist:
        return best
    # fallback: keep expected, unknown r
    return (ex, ey, max_dist / 2.2)


# ==============================================================================
# Training from Answer Key
# ==============================================================================

def learn_from_answer_key(key_bgr: np.ndarray, dp: DetectParams) -> Tuple[GridModel, Dict, Dict]:
    h, w = key_bgr.shape[:2]
    circles = detect_circles_auto(key_bgr, dp)

    if len(circles) < 40:
        raise ValueError("عدد الفقاعات المكتشفة قليل جداً. ارفع DPI أو تأكد وضوح الصفحة وكاملها.")

    # ID grid (top-right)
    id_col_x, id_row_y, id_r, id_dbg = find_id_grid(circles, w, h, id_digits=4, id_rows=10)

    # Q grid (variable)
    q_col_x, q_row_y, q_r, q_dbg = find_question_grid(circles, w, h, choices_candidates=(2, 4, 5))

    model = GridModel(width=w, height=h)
    model.id_digits = 4
    model.id_rows = 10
    model.id_col_x = id_col_x
    model.id_row_y = id_row_y
    model.id_r = id_r

    model.q_cols = len(q_col_x)
    model.q_rows = len(q_row_y)
    model.q_col_x = q_col_x
    model.q_row_y = q_row_y
    model.q_r = q_r

    model.expected_id_centers = build_centers(id_col_x, id_row_y)
    model.expected_q_centers = build_centers(q_col_x, q_row_y)

    # Extract answer key (from key sheet itself)
    gray = cv2.cvtColor(key_bgr, cv2.COLOR_BGR2GRAY)
    choices = list("ABCDE")[:model.q_cols]

    answer_key: Dict[int, str] = {}
    per_q_debug: Dict[int, Dict] = {}

    # map centers per row
    for qi, ry in enumerate(model.q_row_y, start=1):
        fills = []
        xscores = []
        for cx in model.q_col_x:
            # snap to nearest detected circle
            x, y, rr = nearest_circle_center((cx, ry), circles, max_dist=2.5 * model.q_r)
            fill, xscore = fill_ratio_and_xscore(gray, x, y, rr, dp)
            fills.append(fill)
            xscores.append(xscore)

        res = pick_answer_for_row(fills, xscores, choices, dp)
        # even in answer key: if DOUBLE/BLANK, keep as '?' but show debug
        if res["status"] == "OK":
            answer_key[qi] = res["answer"]
        else:
            answer_key[qi] = "?"
        per_q_debug[qi] = res

    dbg = {"id": id_dbg, "q": q_dbg, "total_circles": len(circles)}
    return model, answer_key, {"dbg": dbg, "per_q": per_q_debug, "circles": circles}


# ==============================================================================
# ID & Answers extraction for student sheets
# ==============================================================================

def extract_student_id(bgr: np.ndarray, model: GridModel, dp: DetectParams) -> str:
    bgr = resize_to(bgr, model.width, model.height)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    circles = detect_circles_auto(bgr, dp)

    digits = []
    # for each digit column, pick row 0..9 with max fill
    for col_idx, cx in enumerate(model.id_col_x):
        fills = []
        for ry in model.id_row_y:
            x, y, rr = nearest_circle_center((cx, ry), circles, max_dist=2.6 * model.id_r)
            fill, _xscore = fill_ratio_and_xscore(gray, x, y, rr, dp)
            fills.append(fill)

        top_row = int(np.argmax(np.array(fills))) if fills else -1
        top_fill = fills[top_row] if top_row >= 0 else 0.0

        if top_fill < dp.blank_fill_threshold:
            digits.append("X")
        else:
            digits.append(str(top_row))

    return "".join(digits)


def extract_student_answers(bgr: np.ndarray, model: GridModel, dp: DetectParams) -> Dict[int, Dict]:
    bgr = resize_to(bgr, model.width, model.height)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    circles = detect_circles_auto(bgr, dp)

    choices = list("ABCDE")[:model.q_cols]
    answers: Dict[int, Dict] = {}

    for qi, ry in enumerate(model.q_row_y, start=1):
        fills = []
        xscores = []
        for cx in model.q_col_x:
            x, y, rr = nearest_circle_center((cx, ry), circles, max_dist=2.6 * model.q_r)
            fill, xscore = fill_ratio_and_xscore(gray, x, y, rr, dp)
            fills.append(fill)
            xscores.append(xscore)

        res = pick_answer_for_row(fills, xscores, choices, dp)
        answers[qi] = res

    return answers


def grade_one_sheet(student_bgr: np.ndarray, model: GridModel,
                    answer_key: Dict[int, str], roster: Dict[str, str],
                    strict: bool, dp: DetectParams) -> Dict:
    student_bgr = resize_to(student_bgr, model.width, model.height)
    sid = extract_student_id(student_bgr, model, dp)
    name = roster.get(str(sid).strip(), "غير موجود")

    ans = extract_student_answers(student_bgr, model, dp)

    total = len(answer_key)
    correct = 0
    for q, k in answer_key.items():
        if q not in ans:
            continue
        if strict and ans[q]["status"] != "OK":
            continue
        if ans[q]["answer"] == k:
            correct += 1

    pct = (correct / total * 100.0) if total > 0 else 0.0
    return {
        "student_id": sid,
        "student_name": name,
        "score": correct,
        "total": total,
        "percentage": pct
    }


# ==============================================================================
# Debug overlay
# ==============================================================================

def overlay_grid(img_bgr: np.ndarray, model: GridModel, circles: Optional[List[Tuple[float, float, float]]] = None) -> np.ndarray:
    out = img_bgr.copy()

    # draw detected circles (optional)
    if circles:
        for (x, y, r) in circles:
            cv2.circle(out, (int(x), int(y)), int(r), (0, 255, 255), 1)

    # ID grid lines (RED)
    for cx in model.id_col_x:
        cv2.line(out, (int(cx), 0), (int(cx), model.height), (0, 0, 255), 2)
    for ry in model.id_row_y:
        cv2.circle(out, (int(model.id_col_x[0]), int(ry)), 3, (0, 0, 255), -1)

    # Q grid lines (BLUE)
    for cx in model.q_col_x:
        cv2.line(out, (int(cx), 0), (int(cx), model.height), (255, 0, 0), 2)
    for ry in model.q_row_y:
        cv2.circle(out, (int(model.q_col_x[0]), int(ry)), 3, (255, 0, 0), -1)

    return out


def bgr_to_rgb_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ==============================================================================
# Streamlit App
# ==============================================================================

def read_roster(uploaded) -> Dict[str, str]:
    if uploaded is None:
        return {}
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # expected: student_code, student_name
    if "student_code" not in df.columns or "student_name" not in df.columns:
        raise ValueError("ملف الطلاب يجب أن يحتوي عمودين: student_code و student_name")

    roster = dict(zip(df["student_code"].astype(str).str.strip(),
                      df["student_name"].astype(str).str.strip()))
    return roster


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return buf.getvalue()


def main():
    st.set_page_config(page_title="Hybrid OMR + AI (Robust)", layout="wide")

    st.title("✅ Hybrid OMR + قواعد ذكية (2/4/5 خيارات + عدد أسئلة متغير)")
    st.caption("يعرض Answer Key المستخرج قبل التصحيح للتأكد — ويعالج (X) إلغاء الخيار تلقائياً.")

    # session
    if "model" not in st.session_state:
        st.session_state.model = None
    if "answer_key" not in st.session_state:
        st.session_state.answer_key = None
    if "train_debug" not in st.session_state:
        st.session_state.train_debug = None
    if "key_overlay" not in st.session_state:
        st.session_state.key_overlay = None

    # params
    dp = DetectParams()

    with st.sidebar:
        st.header("⚙️ إعدادات الذكاء (ثابتة)")
        dp.blank_fill_threshold = st.slider("Blank fill threshold", 0.05, 0.40, float(dp.blank_fill_threshold), 0.01)
        dp.double_ratio = st.slider("Double ratio", 1.05, 2.00, float(dp.double_ratio), 0.01)
        dp.x_threshold = st.slider("X cancel threshold", 0.2, 3.0, float(dp.x_threshold), 0.05)
        dp.min_circularity = st.slider("Min circularity", 0.30, 0.85, float(dp.min_circularity), 0.01)
        dp.min_area = int(st.slider("Min area", 80, 2000, int(dp.min_area), 10))
        dp.max_area = int(st.slider("Max area", 3000, 30000, int(dp.max_area), 100))

        st.divider()
        st.info("ملاحظة: بعد ما تثبت الإعدادات وتشتغل، لا تغيّرها إلا إذا تغير نوع الورقة/الطباعة.")

    # =========================
    # Step 1: Upload Answer Key
    # =========================
    st.subheader("1) تدريب/استخراج القالب من Answer Key")
    col1, col2 = st.columns([2, 1])

    with col1:
        key_file = st.file_uploader("📌 ارفع Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"], key="key_uploader")
    with col2:
        dpi = st.selectbox("DPI للـ PDF", [200, 250, 300, 350, 400], index=2)

    train_btn = st.button("🧠 تدريب/استخراج من Answer Key", type="primary", disabled=(key_file is None))

    if train_btn:
        try:
            pil = load_first_page(key_file.getvalue(), key_file.name, dpi=int(dpi))
            key_bgr = to_bgr(pil)

            with st.spinner("جاري التدريب/استخراج الشبكة..."):
                model, answer_key, train_pack = learn_from_answer_key(key_bgr, dp)

            st.session_state.model = model
            st.session_state.answer_key = answer_key
            st.session_state.train_debug = train_pack

            overlay = overlay_grid(resize_to(key_bgr, model.width, model.height), model, circles=train_pack["circles"])
            st.session_state.key_overlay = bgr_to_rgb_pil(overlay)

            st.success("✅ التدريب نجح.")
        except Exception as e:
            st.error(f"❌ فشل التدريب: {e}")
            st.stop()

    # Show training results if available
    if st.session_state.model is not None:
        model: GridModel = st.session_state.model
        answer_key: Dict[int, str] = st.session_state.answer_key

        st.markdown("---")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("عدد الأسئلة المكتشف", int(model.q_rows))
        cB.metric("عدد الخيارات المكتشف", int(model.q_cols))
        cC.metric("خانات كود الطالب", int(model.id_digits))
        cD.metric("صفوف كود الطالب", int(model.id_rows))

        st.subheader("🔑 Answer Key المستخرج (راجع قبل التصحيح)")
        # show as table
        df_key = pd.DataFrame([{"Question": i, "Key": answer_key[i]} for i in sorted(answer_key.keys())])
        st.dataframe(df_key, width="stretch", height=260)

        with st.expander("عرض JSON"):
            st.json({str(k): v for k, v in answer_key.items()})

        st.subheader("🖼️ Overlay للتأكد (أحمر=كود، أزرق=أعمدة الخيارات)")
        if st.session_state.key_overlay is not None:
            st.image(st.session_state.key_overlay, width=None)

        with st.expander("Debug تفاصيل التدريب"):
            st.write(st.session_state.train_debug["dbg"])

    # =========================
    # Step 2: Grading
    # =========================
    st.markdown("---")
    st.subheader("2) التصحيح (Roster + أوراق الطلاب)")

    roster_file = st.file_uploader("📋 ملف الطلاب (Excel/CSV) يجب يحتوي student_code و student_name",
                                   type=["xlsx", "xls", "csv"], key="roster_uploader")
    sheets_files = st.file_uploader("📚 أوراق الطلاب (PDF/صورة) - يمكن تحديد أكثر من ملف",
                                    type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True, key="sheets_uploader")

    strict = st.checkbox("وضع صارم: BLANK/DOUBLE لا تُحسب", value=True)

    grade_btn = st.button("✅ ابدأ التصحيح", disabled=(st.session_state.model is None or roster_file is None or not sheets_files))

    if grade_btn:
        try:
            roster = read_roster(roster_file)
            model: GridModel = st.session_state.model
            answer_key: Dict[int, str] = st.session_state.answer_key

            results = []
            with st.spinner("جاري تصحيح جميع الأوراق..."):
                for f in sheets_files:
                    pil = load_first_page(f.getvalue(), f.name, dpi=int(dpi))
                    bgr = to_bgr(pil)

                    res = grade_one_sheet(bgr, model, answer_key, roster, strict, dp)
                    results.append(res)

            df = pd.DataFrame(results)
            df.insert(0, "status", np.where(df["percentage"] >= 50, "ناجح ✓", "راسب ✗"))
            df["percentage"] = df["percentage"].map(lambda x: f"{x:.1f}%")

            st.success("✅ اكتمل التصحيح.")
            st.dataframe(df, width="stretch", height=320)

            excel_bytes = df_to_excel_bytes(df)
            st.download_button(
                "⬇️ تحميل النتائج Excel",
                data=excel_bytes,
                file_name="results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"❌ خطأ أثناء التصحيح: {e}")
            st.stop()


if __name__ == "__main__":
    main()
