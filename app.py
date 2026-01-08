# app.py
# ============================================================
# ✅ AUTO OMR (NO SLIDERS)
# ✅ يتدرّب على أي Answer Key ترفعه
# ✅ يكتشف تلقائياً:
#    - شبكة كود الطالب (10 صفوف + عدد خانات تلقائياً)
#    - شبكة الأسئلة (عدد الأسئلة + عدد الخيارات تلقائياً)
# ✅ يعرض للمستخدم قبل التصحيح:
#    - عدد الأسئلة المكتشف
#    - Answer Key المستخرج (JSON)
#    - جدول Debug
# ✅ سياسة الشطب:
#    "لو شطب خيار واحد + ظلّل خيار آخر → اختر المظلّل فقط"
# ============================================================

import io
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image


# ==============================
# Safe scalar converters (fix your error)
# ==============================
def safe_float(x) -> float:
    """Convert numpy scalar/array to python float safely."""
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        x = x.ravel()
        if x.size == 0:
            return 0.0
        return float(x[0])
    try:
        return float(x)
    except Exception:
        return 0.0


def safe_int(x) -> int:
    """Convert numpy scalar/array to python int safely."""
    return int(round(safe_float(x)))


# ==============================
# File + image helpers
# ==============================
def read_bytes(uploaded_file) -> bytes:
    if uploaded_file is None:
        return b""
    try:
        return uploaded_file.getbuffer().tobytes()
    except Exception:
        try:
            return uploaded_file.read()
        except Exception:
            return b""


def load_pages(file_bytes: bytes, filename: str, dpi: int = 300) -> List[Image.Image]:
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ==============================
# Models
# ==============================
@dataclass
class BubbleGrid:
    row_y: np.ndarray          # (R,)
    col_x: np.ndarray          # (C,)
    centers: np.ndarray        # (R, C, 2)
    rows: int
    cols: int


@dataclass
class LearnedTemplate:
    ref_bgr: np.ndarray
    ref_w: int
    ref_h: int
    id_grid: BubbleGrid
    q_grid: BubbleGrid
    num_q: int
    num_choices: int
    id_rows: int
    id_digits: int


# ==============================
# Preprocess
# ==============================
def preprocess_binary(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )
    bin_img = cv2.medianBlur(bin_img, 3)
    return bin_img


# ==============================
# Robust bubble center detection (AUTO)
# ==============================
def find_bubble_centers(bin_img: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((0, 2), dtype=np.int32)

    records = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 15:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 10:
            continue
        circ = 4.0 * np.pi * area / (peri * peri + 1e-6)
        x, y, w, h = cv2.boundingRect(c)
        ar = w / (h + 1e-6)
        records.append((area, circ, ar, c))

    if not records:
        return np.zeros((0, 2), dtype=np.int32)

    # learn typical bubble sizes from histogram peak on log(area)
    areas = np.array([r[0] for r in records], dtype=np.float32)
    loga = np.log(np.clip(areas, 1, None))
    hist, edges = np.histogram(loga, bins=25)
    peak = int(np.argmax(hist))
    lo = safe_float(edges[peak])
    hi = safe_float(edges[min(peak + 1, len(edges) - 1)])

    area_lo = float(np.exp(lo)) * 0.5
    area_hi = float(np.exp(hi)) * 2.0

    centers = []
    for area, circ, ar, c in records:
        if not (area_lo <= area <= area_hi):
            continue
        if circ < 0.20:
            continue
        if not (0.50 <= ar <= 2.00):
            continue
        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6:
            continue
        cx = safe_int(M["m10"] / M["m00"])
        cy = safe_int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    return np.array(centers, dtype=np.int32) if centers else np.zeros((0, 2), dtype=np.int32)


# ==============================
# Auto grouping (no sklearn)
# ==============================
def robust_gap_tolerance(sorted_vals: np.ndarray) -> float:
    if len(sorted_vals) < 6:
        return 10.0
    diffs = np.diff(sorted_vals)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 10.0
    p10 = np.percentile(diffs, 10)
    p40 = np.percentile(diffs, 40)
    small = diffs[(diffs >= p10) & (diffs <= p40)]
    base = safe_float(np.median(small)) if len(small) else safe_float(np.median(diffs))
    return max(6.0, base * 0.65)


def group_1d_positions(values: np.ndarray) -> np.ndarray:
    v = np.sort(values.astype(np.float32))
    if len(v) == 0:
        return np.array([], dtype=np.float32)
    tol = robust_gap_tolerance(v)
    groups = []
    cur = [v[0]]
    for x in v[1:]:
        if (x - cur[-1]) <= tol:
            cur.append(x)
        else:
            groups.append(cur)
            cur = [x]
    groups.append(cur)
    centers = np.array([safe_float(np.mean(g)) for g in groups], dtype=np.float32)
    return np.sort(centers)


def snap_to_grid(centers_xy: np.ndarray, row_y: np.ndarray, col_x: np.ndarray) -> np.ndarray:
    R = len(row_y)
    C = len(col_x)
    grid = np.zeros((R, C, 2), dtype=np.float32)
    cnt = np.zeros((R, C), dtype=np.int32)

    for xy in centers_xy:
        x = safe_float(xy[0])
        y = safe_float(xy[1])
        ri = int(np.argmin(np.abs(row_y - y)))
        ci = int(np.argmin(np.abs(col_x - x)))
        grid[ri, ci, 0] += x
        grid[ri, ci, 1] += y
        cnt[ri, ci] += 1

    for r in range(R):
        for c in range(C):
            if cnt[r, c] > 0:
                grid[r, c] /= cnt[r, c]
            else:
                grid[r, c] = (col_x[c], row_y[r])

    return grid


# ==============================
# Split regions (heuristic)
# ==============================
def split_regions(centers: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = centers[:, 0]
    ys = centers[:, 1]

    id_mask = (xs > 0.55 * w) & (ys < 0.60 * h)
    q_mask = (ys > 0.25 * h) & (xs < 0.85 * w)

    id_c = centers[id_mask]
    q_c = centers[q_mask]

    # fallback
    if id_c.shape[0] < 20:
        id_c = centers[(xs > np.median(xs)) & (ys < np.median(ys))]
    if q_c.shape[0] < 25:
        q_c = centers[(ys > np.median(ys))]

    return id_c, q_c


# ==============================
# Choose best 4-6 columns for questions (robust)
# ==============================
def select_best_columns(q_col_x: np.ndarray, min_choices: int = 4, max_choices: int = 6) -> np.ndarray:
    """
    If many columns detected, choose the tightest consecutive window of size 4..6
    (the real choices are typically tightly spaced compared to stray columns).
    """
    q_col_x = np.sort(q_col_x.astype(np.float32))
    n = len(q_col_x)
    if n <= max_choices:
        return q_col_x
    best = None
    best_span = 1e18
    for k in range(min_choices, max_choices + 1):
        if n < k:
            continue
        for i in range(0, n - k + 1):
            span = safe_float(q_col_x[i + k - 1] - q_col_x[i])
            if span < best_span:
                best_span = span
                best = q_col_x[i:i + k]
    if best is None:
        return q_col_x[:min_choices]
    return np.array(best, dtype=np.float32)


# ==============================
# Alignment (student -> key)
# ==============================
def orb_align(student_bgr: np.ndarray, ref_bgr: np.ndarray) -> Tuple[np.ndarray, bool, int]:
    h, w = ref_bgr.shape[:2]
    orb = cv2.ORB_create(5000)

    g1 = cv2.cvtColor(student_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)

    if d1 is None or d2 is None or len(k1) < 25 or len(k2) < 25:
        return cv2.resize(student_bgr, (w, h)), False, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 25:
        return cv2.resize(student_bgr, (w, h)), False, len(good)

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return cv2.resize(student_bgr, (w, h)), False, len(good)

    warped = cv2.warpPerspective(student_bgr, H, (w, h))
    return warped, True, len(good)


# ==============================
# Bubble scoring (Fill + X)
# ==============================
def bubble_roi(gray: np.ndarray, cx: int, cy: int, win: int = 18) -> np.ndarray:
    h, w = gray.shape[:2]
    x1 = max(0, cx - win); x2 = min(w, cx + win)
    y1 = max(0, cy - win); y2 = min(h, cy + win)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    rh, rw = roi.shape
    mh = int(rh * 0.18)
    mw = int(rw * 0.18)
    inner = roi[mh:rh - mh, mw:rw - mw]
    return inner if inner.size > 0 else roi


def fill_score(roi_gray: np.ndarray) -> float:
    if roi_gray.size < 10:
        return 0.0
    g = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return safe_float(np.mean(th > 0))


def x_mark_score(roi_gray: np.ndarray) -> float:
    g = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    edges = cv2.Canny(g, 50, 150)
    min_len = max(10, int(min(roi_gray.shape) * 0.45))
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=18, minLineLength=min_len, maxLineGap=3
    )
    if lines is None:
        return 0.0
    total = 0.0
    for x1, y1, x2, y2 in lines[:, 0]:
        total += float(np.hypot(x2 - x1, y2 - y1))
    norm = float(roi_gray.shape[0] + roi_gray.shape[1]) + 1e-9
    return total / norm


def bubble_stats(gray: np.ndarray, cx: int, cy: int, win: int = 18) -> dict:
    roi = bubble_roi(gray, cx, cy, win=win)
    return {
        "fill": float(fill_score(roi)),
        "std": float(np.std(roi)),
        "xscore": float(x_mark_score(roi)),
    }


def auto_threshold_from_key(all_fills: np.ndarray) -> Tuple[float, float]:
    fills = np.clip(all_fills, 0.0, 1.0)
    x = (fills * 255.0).astype(np.uint8).reshape(-1, 1)
    _, thr = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blank = safe_float(thr) / 255.0
    blank = max(0.05, min(0.25, blank * 0.6))
    double_gap = 0.08
    return blank, double_gap


def pick_choice_x_cancel(stats_list: List[dict],
                         labels: List[str],
                         blank_fill_thresh: float,
                         double_gap: float,
                         x_std_thresh: float,
                         x_score_thresh: float) -> Tuple[str, str, List[bool], List[float]]:
    cancelled = []
    fills = []
    for s in stats_list:
        fills.append(float(s["fill"]))
        cancelled.append((float(s["std"]) >= x_std_thresh) and (float(s["xscore"]) >= x_score_thresh))

    candidates = [i for i in range(len(labels)) if not cancelled[i]]
    if not candidates:
        return "?", "CANCELLED_ALL", cancelled, fills

    candidates.sort(key=lambda i: fills[i], reverse=True)
    best = candidates[0]
    best_f = fills[best]
    second_f = fills[candidates[1]] if len(candidates) > 1 else 0.0

    if best_f < blank_fill_thresh:
        return "?", "BLANK", cancelled, fills
    if (best_f - second_f) < double_gap:
        return "!", "DOUBLE", cancelled, fills

    return labels[best], "OK", cancelled, fills


# ==============================
# Roster
# ==============================
def load_roster(file, id_digits: int) -> Dict[str, str]:
    if file.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "student_code" not in df.columns or "student_name" not in df.columns:
        raise ValueError("ملف الطلاب لازم يحتوي: student_code و student_name")

    codes = (
        df["student_code"]
        .astype(str).str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(id_digits)
    )
    names = df["student_name"].astype(str).str.strip()
    return dict(zip(codes, names))


# ==============================
# Learn template from Answer Key (AUTO)
# ==============================
def learn_template_from_key(key_bgr: np.ndarray) -> Tuple[LearnedTemplate, dict]:
    h, w = key_bgr.shape[:2]
    bin_img = preprocess_binary(key_bgr)
    centers = find_bubble_centers(bin_img)

    if centers.shape[0] < 20:
        raise ValueError("لم يتم العثور على عناصر كافية. جرّب DPI أعلى أو صورة أوضح.")

    id_centers, q_centers = split_regions(centers, w, h)

    if id_centers.shape[0] < 20:
        raise ValueError("فشل اكتشاف منطقة كود الطالب تلقائياً. تأكد الكود ظاهر بوضوح.")
    if q_centers.shape[0] < 20:
        raise ValueError("فشل اكتشاف منطقة الأسئلة تلقائياً. تأكد الأسئلة ظاهرة بوضوح.")

    # ID grid
    id_row_y = group_1d_positions(id_centers[:, 1])
    id_col_x = group_1d_positions(id_centers[:, 0])

    if len(id_row_y) < 8 or len(id_col_x) < 2:
        raise ValueError("تعذر بناء شبكة الكود. صورة الكود غير واضحة أو مقصوصة.")

    # Normalize rows toward 10 if close
    if 9 <= len(id_row_y) <= 11:
        while len(id_row_y) > 10:
            dif = np.diff(id_row_y)
            j = int(np.argmin(dif))
            merged = (id_row_y[j] + id_row_y[j + 1]) / 2.0
            id_row_y = np.delete(id_row_y, [j, j + 1])
            id_row_y = np.sort(np.append(id_row_y, merged))
        while len(id_row_y) < 10:
            step = safe_float(np.median(np.diff(id_row_y)))
            if not np.isfinite(step) or step <= 0:
                step = 20.0
            id_row_y = np.sort(np.append(id_row_y, id_row_y[-1] + step))

    id_digits = int(len(id_col_x))
    id_grid_centers = snap_to_grid(id_centers, id_row_y, id_col_x)
    id_grid = BubbleGrid(row_y=id_row_y, col_x=id_col_x, centers=id_grid_centers, rows=len(id_row_y), cols=len(id_col_x))

    # Q grid
    q_row_y = group_1d_positions(q_centers[:, 1])
    q_col_x = group_1d_positions(q_centers[:, 0])
    q_col_x = select_best_columns(q_col_x, 4, 6)

    if len(q_col_x) < 4:
        raise ValueError("لم يتم اكتشاف أعمدة الخيارات (A-D). ربما الصفحة مقصوصة أو غير واضحة.")

    num_choices = int(len(q_col_x))
    num_q = int(len(q_row_y))

    q_grid_centers = snap_to_grid(q_centers, q_row_y, q_col_x)
    q_grid = BubbleGrid(row_y=q_row_y, col_x=q_col_x, centers=q_grid_centers, rows=num_q, cols=num_choices)

    template = LearnedTemplate(
        ref_bgr=key_bgr, ref_w=w, ref_h=h,
        id_grid=id_grid, q_grid=q_grid,
        num_q=num_q, num_choices=num_choices,
        id_rows=10, id_digits=id_digits
    )

    dbg = {
        "centers_total": int(centers.shape[0]),
        "id_centers": int(id_centers.shape[0]),
        "q_centers": int(q_centers.shape[0]),
        "id_rows": int(len(id_row_y)),
        "id_digits": int(len(id_col_x)),
        "q_rows": int(len(q_row_y)),
        "q_cols": int(len(q_col_x)),
    }
    return template, dbg


# ==============================
# Extract Answer Key (AUTO thresholds)
# ==============================
def extract_answer_key(template: LearnedTemplate) -> Tuple[Dict[int, str], pd.DataFrame, dict]:
    gray = cv2.cvtColor(template.ref_bgr, cv2.COLOR_BGR2GRAY)
    choices = list("ABCDEFGH"[:template.num_choices])

    all_fills = []
    cache = []
    for r in range(template.q_grid.rows):
        row_stats = []
        for c in range(template.q_grid.cols):
            cx, cy = template.q_grid.centers[r, c]
            stt = bubble_stats(gray, safe_int(cx), safe_int(cy), win=18)
            row_stats.append(stt)
            all_fills.append(stt["fill"])
        cache.append(row_stats)

    blank_fill, double_gap = auto_threshold_from_key(np.array(all_fills, dtype=np.float32))

    # X thresholds for KEY
    x_std_key = 18.0
    x_score_key = 0.90

    key = {}
    dbg_rows = []

    for r in range(template.q_grid.rows):
        stats = cache[r]
        ans, status, cancelled, fills = pick_choice_x_cancel(
            stats, choices,
            blank_fill_thresh=blank_fill,
            double_gap=double_gap,
            x_std_thresh=x_std_key,
            x_score_thresh=x_score_key
        )

        if status == "OK":
            key[r + 1] = ans

        dbg_rows.append(
            [r + 1, ans, status, cancelled] +
            [round(float(s["fill"]), 3) for s in stats] +
            [round(float(s["std"]), 1) for s in stats] +
            [round(float(s["xscore"]), 2) for s in stats]
        )

    df_dbg = pd.DataFrame(
        dbg_rows,
        columns=(["Q", "Picked", "Status", "CancelledFlags"] +
                 [f"fill_{c}" for c in choices] +
                 [f"std_{c}" for c in choices] +
                 [f"x_{c}" for c in choices])
    )

    params = {
        "blank_fill_thresh": float(blank_fill),
        "double_gap": float(double_gap),
        "choices": choices,
    }
    return key, df_dbg, params


# ==============================
# Read student ID and answers
# ==============================
def read_student_id(template: LearnedTemplate, gray: np.ndarray, blank_fill: float, double_gap: float) -> str:
    labels = [str(i) for i in range(template.id_grid.rows)]
    digits = []

    for c in range(template.id_grid.cols):
        fills = []
        for r in range(template.id_grid.rows):
            cx, cy = template.id_grid.centers[r, c]
            roi = bubble_roi(gray, safe_int(cx), safe_int(cy), win=16)
            fills.append(fill_score(roi))

        idx = np.argsort(fills)[::-1]
        best = int(idx[0])
        second = int(idx[1]) if len(idx) > 1 else best

        best_f = float(fills[best])
        second_f = float(fills[second])

        if best_f < blank_fill:
            digit = "X"
        elif (best_f - second_f) < double_gap:
            digit = "X"
        else:
            digit = labels[best]

        digits.append(digit)

    return "".join(digits)


def read_student_answers(template: LearnedTemplate,
                         gray: np.ndarray,
                         choices: List[str],
                         blank_fill: float,
                         double_gap: float) -> pd.DataFrame:
    # X thresholds for STUDENTS (usually stronger scribbles)
    x_std_student = 22.0
    x_score_student = 1.2

    rows = []
    for r in range(template.q_grid.rows):
        stats = []
        for c in range(template.q_grid.cols):
            cx, cy = template.q_grid.centers[r, c]
            stats.append(bubble_stats(gray, safe_int(cx), safe_int(cy), win=18))

        ans, status, cancelled, _ = pick_choice_x_cancel(
            stats, choices,
            blank_fill_thresh=blank_fill,
            double_gap=double_gap,
            x_std_thresh=x_std_student,
            x_score_thresh=x_score_student
        )

        rows.append([r + 1, ans, status, cancelled])

    return pd.DataFrame(rows, columns=["Q", "Picked", "Status", "CancelledFlags"])


# ==============================
# Streamlit App
# ==============================
def main():
    st.set_page_config(page_title="AUTO OMR - Any Answer Key", layout="wide")
    st.title("✅ AUTO OMR: تدريب تلقائي على أي Answer Key + عرض الأنسر قبل التصحيح")

    if "trained" not in st.session_state:
        st.session_state.trained = False
        st.session_state.template = None
        st.session_state.answer_key = None
        st.session_state.key_dbg = None
        st.session_state.params = None
        st.session_state.preview_img = None
        st.session_state.train_dbg = None

    col1, col2, col3 = st.columns(3)
    with col1:
        roster_file = st.file_uploader("📋 ملف الطلاب (Excel/CSV)", type=["xlsx", "xls", "csv"])
    with col2:
        key_file = st.file_uploader("🔑 Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    with col3:
        sheets_file = st.file_uploader("📚 أوراق الطلاب (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])

    dpi = st.selectbox("DPI للـ PDF", [200, 250, 300, 350, 400], index=2)
    debug = st.checkbox("Debug (عرض تفاصيل)", value=True)

    if not key_file:
        st.info("ارفع Answer Key أولاً ليبدأ التدريب.")
        return

    # Train / retrain
    if (not st.session_state.trained) or st.button("🔄 إعادة التدريب من الأنسر"):
        try:
            key_pages = load_pages(read_bytes(key_file), key_file.name, dpi=int(dpi))
            if not key_pages:
                st.error("فشل قراءة الأنسر")
                return

            key_bgr = pil_to_bgr(key_pages[0])

            template, train_dbg = learn_template_from_key(key_bgr)
            answer_key, df_key_dbg, params = extract_answer_key(template)

            # Preview with points
            vis = key_bgr.copy()
            # questions
            for r in range(template.q_grid.rows):
                for c in range(template.q_grid.cols):
                    x, y = template.q_grid.centers[r, c]
                    cv2.circle(vis, (safe_int(x), safe_int(y)), 6, (0, 255, 0), 2)
            # id
            for r in range(template.id_grid.rows):
                for c in range(template.id_grid.cols):
                    x, y = template.id_grid.centers[r, c]
                    cv2.circle(vis, (safe_int(x), safe_int(y)), 6, (0, 0, 255), 2)

            st.session_state.template = template
            st.session_state.answer_key = answer_key
            st.session_state.key_dbg = df_key_dbg
            st.session_state.params = params
            st.session_state.preview_img = vis
            st.session_state.train_dbg = train_dbg
            st.session_state.trained = True

            st.success("✅ تم التدريب تلقائياً من الأنسر.")
        except Exception as e:
            st.error(f"❌ فشل التدريب التلقائي: {e}")
            with st.expander("📌 تفاصيل الخطأ (Traceback)"):
                st.code(traceback.format_exc())
            st.session_state.trained = False
            return

    # Show training results
    template = st.session_state.template
    answer_key = st.session_state.answer_key
    df_key_dbg = st.session_state.key_dbg
    params = st.session_state.params

    st.markdown("---")
    st.subheader("📌 نتيجة التدريب (قبل التصحيح)")

    cA, cB, cC, cD = st.columns(4)
    with cA:
        st.metric("عدد الأسئلة المكتشف", int(template.num_q))
    with cB:
        st.metric("عدد الخيارات المكتشف", int(template.num_choices))
    with cC:
        st.metric("خانات كود الطالب المكتشفة", int(template.id_digits))
    with cD:
        st.metric("صفوف كود الطالب المكتشفة", int(template.id_grid.rows))

    if st.session_state.train_dbg:
        st.caption(f"Detected centers: {st.session_state.train_dbg}")

    st.markdown("### 🔑 Answer Key المستخرج")
    st.json(answer_key)

    st.info(f"Auto thresholds: blank_fill={params['blank_fill_thresh']:.3f}, double_gap={params['double_gap']:.3f}")

    if debug:
        st.markdown("### Debug Table (مهم جداً: تأكد من Q5/Q6 هنا)")
        st.dataframe(df_key_dbg, width="stretch", height=420)
        st.image(bgr_to_rgb(st.session_state.preview_img), caption="Green=Questions, Red=ID", width="stretch")

    st.markdown("---")
    st.subheader("✅ التصحيح")

    if not roster_file or not sheets_file:
        st.warning("ارفع ملف الطلاب + أوراق الطلاب للبدء.")
        return

    # Load roster
    try:
        roster = load_roster(roster_file, id_digits=int(template.id_digits))
        st.success(f"✅ تم تحميل قائمة الطلاب: {len(roster)}")
    except Exception as e:
        st.error(f"خطأ ملف الطلاب: {e}")
        return

    if st.button("🚀 ابدأ التصحيح الآن", type="primary"):
        try:
            pages = load_pages(read_bytes(sheets_file), sheets_file.name, dpi=int(dpi))
            if not pages:
                st.error("فشل قراءة أوراق الطلاب")
                return

            blank_fill = float(params["blank_fill_thresh"])
            double_gap = float(params["double_gap"])
            choices = params["choices"]

            results = []
            prog = st.progress(0)

            for i, pil_page in enumerate(pages, start=1):
                page_bgr = pil_to_bgr(pil_page)

                aligned, ok, good = orb_align(page_bgr, template.ref_bgr)
                gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

                code = read_student_id(template, gray, blank_fill, double_gap)
                code = str(code).zfill(int(template.id_digits))
                name = roster.get(code, "غير موجود")

                df_ans = read_student_answers(template, gray, choices, blank_fill, double_gap)

                correct = 0
                total = len(answer_key)

                for _, row in df_ans.iterrows():
                    q = int(row["Q"])
                    if q not in answer_key:
                        continue
                    if row["Status"] != "OK":
                        continue
                    if row["Picked"] == answer_key[q]:
                        correct += 1

                pct = (correct / total * 100.0) if total else 0.0
                results.append({
                    "page": i,
                    "aligned_ok": ok,
                    "good_matches": good,
                    "student_code": code,
                    "student_name": name,
                    "score": correct,
                    "total": total,
                    "percentage": round(pct, 2),
                    "status": "ناجح ✓" if pct >= 50 else "راسب ✗",
                })

                prog.progress(int(i / len(pages) * 100))

            df_res = pd.DataFrame(results)
            st.success("✅ اكتمل التصحيح")
            st.dataframe(df_res, width="stretch", height=420)

            out = io.BytesIO()
            df_res.to_excel(out, index=False, engine="openpyxl")
            st.download_button(
                "⬇️ تحميل النتائج Excel",
                data=out.getvalue(),
                file_name="results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء التصحيح: {e}")
            with st.expander("📌 تفاصيل الخطأ (Traceback)"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
