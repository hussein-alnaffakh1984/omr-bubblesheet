import io
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image


# ============================================================
# Helpers: file reading / conversion
# ============================================================
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


def safe_int(x) -> int:
    try:
        if isinstance(x, np.ndarray):
            x = x.ravel()
            return int(round(float(x[0]))) if x.size else 0
        return int(round(float(x)))
    except Exception:
        return 0


# ============================================================
# Data structures
# ============================================================
@dataclass
class BubbleGrid:
    row_y: np.ndarray      # (R,)
    col_x: np.ndarray      # (C,)
    centers: np.ndarray    # (R,C,2)
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
    id_rows: int = 10
    id_digits: int = 4


# ============================================================
# Preprocessing & bubble centers
# ============================================================
def preprocess_binary(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )
    bin_img = cv2.medianBlur(bin_img, 3)
    return bin_img


def find_bubble_centers(bin_img: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((0, 2), dtype=np.int32)

    rec = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 25:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 25:
            continue
        circ = 4.0 * np.pi * area / (peri * peri + 1e-6)
        x, y, w, h = cv2.boundingRect(c)
        ar = w / (h + 1e-6)
        rec.append((area, circ, ar, c))

    if not rec:
        return np.zeros((0, 2), dtype=np.int32)

    areas = np.array([r[0] for r in rec], dtype=np.float32)
    loga = np.log(np.clip(areas, 1, None))
    hist, edges = np.histogram(loga, bins=25)
    pk = int(np.argmax(hist))
    lo = float(edges[pk])
    hi = float(edges[min(pk + 1, len(edges) - 1)])

    area_lo = float(np.exp(lo)) * 0.45
    area_hi = float(np.exp(hi)) * 3.0

    centers = []
    for area, circ, ar, c in rec:
        if not (area_lo <= area <= area_hi):
            continue
        if circ < 0.35:
            continue
        if not (0.75 <= ar <= 1.35):
            continue
        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    return np.array(centers, dtype=np.int32) if centers else np.zeros((0, 2), dtype=np.int32)


# ============================================================
# 1D grouping / snapping (no sklearn)
# ============================================================
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
    base = float(np.median(small)) if len(small) else float(np.median(diffs))
    return max(6.0, base * 0.70)


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
    centers = np.array([float(np.mean(g)) for g in groups], dtype=np.float32)
    return np.sort(centers)


def select_best_columns(col_x: np.ndarray, k: int) -> np.ndarray:
    col_x = np.sort(col_x.astype(np.float32))
    n = len(col_x)
    if n <= k:
        return col_x
    best = None
    best_score = 1e18
    for i in range(0, n - k + 1):
        win = col_x[i:i+k]
        span = float(win[-1] - win[0])
        dif = np.diff(win)
        sstd = float(np.std(dif)) if len(dif) else 0.0
        score = span + 80.0 * sstd
        if score < best_score:
            best_score = score
            best = win
    return np.array(best, dtype=np.float32) if best is not None else col_x[:k]


def snap_to_grid(centers_xy: np.ndarray, row_y: np.ndarray, col_x: np.ndarray) -> np.ndarray:
    R = len(row_y)
    C = len(col_x)
    grid = np.zeros((R, C, 2), dtype=np.float32)
    cnt = np.zeros((R, C), dtype=np.int32)

    for xy in centers_xy:
        x = float(xy[0]); y = float(xy[1])
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


# ============================================================
# Split regions (ID vs Questions)
# ============================================================
def split_regions(centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs = centers[:, 0].astype(np.float32)
    splits = np.percentile(xs, [30, 35, 40, 45, 50, 55, 60, 65])

    def score_id(block: np.ndarray) -> float:
        if block.shape[0] < 25:
            return -1e9
        ry = group_1d_positions(block[:, 1])
        cx = group_1d_positions(block[:, 0])
        return -abs(len(ry) - 10) * 60 - abs(len(cx) - 4) * 120 + block.shape[0] * 0.2

    def score_q(block: np.ndarray) -> float:
        if block.shape[0] < 25:
            return -1e9
        ry = group_1d_positions(block[:, 1])
        cx = group_1d_positions(block[:, 0])
        return len(ry) * 8 + block.shape[0] * 0.05 - min(abs(len(cx) - k) for k in (2, 4, 5)) * 80

    best = None
    best_score = -1e18

    for s in splits:
        left = centers[centers[:, 0] <= s]
        right = centers[centers[:, 0] > s]

        sc1 = score_id(right) + score_q(left)
        sc2 = score_id(left) + score_q(right)

        if sc1 > best_score:
            best_score = sc1
            best = (right, left)
        if sc2 > best_score:
            best_score = sc2
            best = (left, right)

    return best if best is not None else (centers, centers)


# ============================================================
# Choose columns for choices robustly (avoid ID leakage)
# ============================================================
def sanitize_q_centers(q_centers: np.ndarray) -> np.ndarray:
    qx = q_centers[:, 0].astype(np.float32)
    x_lo, x_hi = np.percentile(qx, 10), np.percentile(qx, 90)
    return q_centers[(qx >= x_lo) & (qx <= x_hi)]


def top_x_peaks(q_centers: np.ndarray, k: int) -> np.ndarray:
    xs = q_centers[:, 0].astype(np.float32)
    clusters = group_1d_positions(xs)
    if len(clusters) == 0:
        return np.array([], dtype=np.float32)

    xs_sort = np.sort(xs)
    dx = np.diff(xs_sort)
    dx = dx[(dx > 2) & (dx < 250)]
    tol_x = max(10.0, float(np.percentile(dx, 15)) * 0.60) if len(dx) else 14.0

    counts = []
    for x0 in clusters:
        cnt = int(np.sum(np.abs(xs - x0) < tol_x))
        counts.append((cnt, x0))
    counts.sort(reverse=True, key=lambda t: t[0])
    picked = [x for _, x in counts[:k]]
    return np.sort(np.array(picked, dtype=np.float32))


def pick_choice_columns(q_centers: np.ndarray) -> np.ndarray:
    q_centers = sanitize_q_centers(q_centers)
    qx = q_centers[:, 0].astype(np.float32)

    candidates = []
    for k in (2, 4, 5):
        cols_k = top_x_peaks(q_centers, k)
        if len(cols_k) != k:
            continue
        dif = np.diff(cols_k)
        if len(dif) >= 2:
            med = float(np.median(dif))
            if np.max(dif) > 1.8 * med:
                continue
        score = 0
        for x0 in cols_k:
            score += int(np.sum(np.abs(qx - x0) < 14))
        candidates.append((score, cols_k))

    if not candidates:
        raise ValueError("لم أستطع تحديد أعمدة الخيارات (2/4/5) بدون خلط مع كود الطالب.")
    candidates.sort(reverse=True, key=lambda t: t[0])
    return candidates[0][1]


def build_question_rows(q_centers: np.ndarray, col_x: np.ndarray) -> np.ndarray:
    row_y_all = group_1d_positions(q_centers[:, 1])
    if len(row_y_all) == 0:
        return row_y_all

    xs = q_centers[:, 0].astype(np.float32)
    ys = q_centers[:, 1].astype(np.float32)

    xs_sort = np.sort(xs); ys_sort = np.sort(ys)
    dx = np.diff(xs_sort); dx = dx[(dx > 2) & (dx < 250)]
    dy = np.diff(ys_sort); dy = dy[(dy > 2) & (dy < 250)]
    tol_x = max(10.0, float(np.percentile(dx, 15)) * 0.60) if len(dx) else 14.0
    tol_y = max(10.0, float(np.percentile(dy, 15)) * 0.60) if len(dy) else 12.0

    k = len(col_x)
    need = max(2, k - 1)

    valid = []
    for y0 in row_y_all:
        near = q_centers[np.abs(q_centers[:, 1] - y0) < tol_y]
        if near.shape[0] < need:
            continue
        hits = 0
        for x0 in col_x:
            if np.any(np.abs(near[:, 0] - x0) < tol_x):
                hits += 1
        if hits >= need:
            valid.append(y0)
    return np.array(valid, dtype=np.float32)


# ============================================================
# Train template from Answer Key
# ============================================================
def learn_template_from_key(key_bgr: np.ndarray) -> Tuple[LearnedTemplate, dict]:
    h, w = key_bgr.shape[:2]
    bin_img = preprocess_binary(key_bgr)
    centers = find_bubble_centers(bin_img)

    if centers.shape[0] < 40:
        raise ValueError("عدد الفقاعات المكتشفة قليل جداً. ارفع DPI أو تأكد وضوح الصورة.")

    id_centers, q_centers = split_regions(centers)
    if id_centers.shape[0] < 25:
        raise ValueError("فشل اكتشاف منطقة كود الطالب تلقائياً.")
    if q_centers.shape[0] < 25:
        raise ValueError("فشل اكتشاف منطقة الأسئلة تلقائياً.")

    # ID: force 10 rows and 4 columns
    id_row_y = group_1d_positions(id_centers[:, 1])
    id_col_x = group_1d_positions(id_centers[:, 0])

    if len(id_row_y) > 10:
        while len(id_row_y) > 10:
            dif = np.diff(id_row_y)
            j = int(np.argmin(dif))
            merged = (id_row_y[j] + id_row_y[j + 1]) / 2.0
            id_row_y = np.delete(id_row_y, [j, j + 1])
            id_row_y = np.sort(np.append(id_row_y, merged))
    elif len(id_row_y) < 10:
        step = float(np.median(np.diff(id_row_y))) if len(id_row_y) >= 2 else 20.0
        while len(id_row_y) < 10:
            id_row_y = np.append(id_row_y, id_row_y[-1] + step)
        id_row_y = np.sort(id_row_y)

    if len(id_col_x) >= 4:
        id_col_x = select_best_columns(id_col_x, 4)
    else:
        raise ValueError("لم يتم اكتشاف 4 أعمدة للكود. تأكد من وضوح منطقة الكود.")

    id_cent = snap_to_grid(id_centers, id_row_y, id_col_x)
    id_grid = BubbleGrid(id_row_y, id_col_x, id_cent, rows=10, cols=4)

    # Questions: pick 2/4/5 robustly
    q_col_x = pick_choice_columns(q_centers)
    num_choices = int(len(q_col_x))

    q_row_y = build_question_rows(q_centers, q_col_x)
    if len(q_row_y) < 3:
        raise ValueError("لم يتم اكتشاف صفوف أسئلة كافية. ربما الصفحة مقصوصة أو غير واضحة.")

    q_cent = snap_to_grid(q_centers, q_row_y, q_col_x)
    q_grid = BubbleGrid(q_row_y, q_col_x, q_cent, rows=len(q_row_y), cols=num_choices)

    template = LearnedTemplate(
        ref_bgr=key_bgr, ref_w=w, ref_h=h,
        id_grid=id_grid,
        q_grid=q_grid,
        num_q=len(q_row_y),
        num_choices=num_choices,
    )

    dbg = {
        "centers_total": int(centers.shape[0]),
        "id_centers": int(id_centers.shape[0]),
        "q_centers": int(q_centers.shape[0]),
        "id_rows": 10,
        "id_digits": 4,
        "q_rows": int(len(q_row_y)),
        "q_cols": int(num_choices),
        "q_col_x": [float(x) for x in q_col_x],
    }
    return template, dbg


# ============================================================
# Bubble scoring: fill + X-cancel detection
# ============================================================
def crop_roi(bgr: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    half = size // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)
    roi = bgr[y1:y2, x1:x2]
    return roi


def fill_ratio_from_roi(roi_bgr: np.ndarray) -> float:
    if roi_bgr.size == 0:
        return 0.0
    bin_roi = preprocess_binary(roi_bgr)
    hh, ww = bin_roi.shape[:2]
    mh = int(hh * 0.25)
    mw = int(ww * 0.25)
    inner = bin_roi[mh:hh-mh, mw:ww-mw]
    if inner.size == 0:
        return 0.0
    return float(np.mean(inner > 0))


def x_cancel_score(roi_bgr: np.ndarray) -> float:
    """
    Detect X mark in the bubble ROI by looking for diagonal line evidence.
    Returns score ~0..1 higher = more likely X.
    """
    if roi_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # edges
    edges = cv2.Canny(gray, 50, 150)
    # Hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25,
                            minLineLength=max(10, roi_bgr.shape[1]//3),
                            maxLineGap=6)
    if lines is None:
        return 0.0

    # count diagonal-ish lines (around 45 or 135 degrees)
    diag = 0
    total = 0
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        ang = np.degrees(np.arctan2(dy, dx + 1e-6))
        ang = (ang + 180) % 180  # 0..180
        total += 1
        if 25 <= ang <= 65 or 115 <= ang <= 155:
            diag += 1

    if total == 0:
        return 0.0
    return float(diag / total)


def detect_choice_for_row(
    bgr: np.ndarray,
    centers_row: np.ndarray,  # (C,2)
    choices: str,
    blank_thr: float,
    double_gap: float,
    x_thr: float
) -> Dict:
    fills = []
    x_scores = []

    for c in range(centers_row.shape[0]):
        x, y = centers_row[c]
        x = safe_int(x); y = safe_int(y)
        roi = crop_roi(bgr, x, y, size=34)
        fills.append(fill_ratio_from_roi(roi))
        x_scores.append(x_cancel_score(roi))

    fills = np.array(fills, dtype=np.float32)
    x_scores = np.array(x_scores, dtype=np.float32)

    # mark cancelled
    cancelled = x_scores >= x_thr

    # prefer non-cancelled
    candidates = np.where(~cancelled)[0]
    if candidates.size == 0:
        # all cancelled -> treat as blank
        return {"answer": "?", "status": "CANCELLED", "fills": fills.tolist(), "x": x_scores.tolist()}

    # choose best fill among non-cancelled
    idx_sorted = candidates[np.argsort(-fills[candidates])]
    top = int(idx_sorted[0])
    top_fill = float(fills[top])
    second_fill = float(fills[idx_sorted[1]]) if idx_sorted.size > 1 else 0.0

    if top_fill < blank_thr:
        return {"answer": "?", "status": "BLANK", "fills": fills.tolist(), "x": x_scores.tolist()}

    # double check only among non-cancelled
    if idx_sorted.size > 1:
        if (top_fill - second_fill) < double_gap:
            return {"answer": "!", "status": "DOUBLE", "fills": fills.tolist(), "x": x_scores.tolist()}

    return {"answer": choices[top], "status": "OK", "fills": fills.tolist(), "x": x_scores.tolist()}


def extract_student_id(bgr: np.ndarray, template: LearnedTemplate, blank_thr_id: float) -> str:
    digits = []
    for c in range(template.id_digits):
        # 10 rows digits 0..9
        vals = []
        for r in range(template.id_rows):
            x, y = template.id_grid.centers[r, c]
            roi = crop_roi(bgr, safe_int(x), safe_int(y), size=30)
            vals.append(fill_ratio_from_roi(roi))
        vals = np.array(vals, dtype=np.float32)
        if float(np.max(vals)) < blank_thr_id:
            digits.append("X")
        else:
            digits.append(str(int(np.argmax(vals))))
    return "".join(digits)


def extract_answers(bgr: np.ndarray, template: LearnedTemplate,
                    blank_thr: float, double_gap: float, x_thr: float) -> Tuple[Dict[int, str], Dict[int, Dict]]:
    choices = "ABCDE"[:template.num_choices]
    ans = {}
    dbg = {}
    for q in range(template.num_q):
        row_cent = template.q_grid.centers[q, :, :]
        res = detect_choice_for_row(bgr, row_cent, choices, blank_thr, double_gap, x_thr)
        dbg[q+1] = res
        ans[q+1] = res["answer"]
    return ans, dbg


# ============================================================
# Build overlay preview
# ============================================================
def make_overlay(template: LearnedTemplate) -> np.ndarray:
    vis = template.ref_bgr.copy()

    # question points (green)
    for r in range(template.q_grid.rows):
        for c in range(template.q_grid.cols):
            x, y = template.q_grid.centers[r, c]
            cv2.circle(vis, (safe_int(x), safe_int(y)), 6, (0, 255, 0), 2)

    # ID points (red)
    for r in range(template.id_grid.rows):
        for c in range(template.id_grid.cols):
            x, y = template.id_grid.centers[r, c]
            cv2.circle(vis, (safe_int(x), safe_int(y)), 6, (0, 0, 255), 2)

    # choice columns (blue)
    for x0 in template.q_grid.col_x:
        cv2.line(vis, (safe_int(x0), 0), (safe_int(x0), template.ref_h), (255, 0, 0), 2)

    return vis


# ============================================================
# Streamlit App
# ============================================================
def main():
    st.set_page_config(page_title="OMR Hybrid (Auto + X-Rule)", layout="wide")
    st.title("✅ OMR ذكي: تدريب تلقائي + عرض Answer Key قبل التصحيح + قاعدة X")

    # --- State
    if "trained" not in st.session_state:
        st.session_state.trained = False
        st.session_state.template = None
        st.session_state.train_dbg = None
        st.session_state.overlay = None
        st.session_state.answer_key = None
        st.session_state.last_error = ""

    # --- Inputs
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        key_file = st.file_uploader("🔑 Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    with c2:
        dpi = st.selectbox("DPI للـ PDF", [200, 250, 300, 350, 400], index=2)
    with c3:
        debug = st.checkbox("Debug", value=True)

    st.markdown("### ⚙️ عتبات الكشف (يمكنك تركها كما هي)")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        blank_thr = st.slider("Blank fill threshold", 0.02, 0.35, 0.14, 0.01)
    with t2:
        double_gap = st.slider("Double gap threshold", 0.00, 0.20, 0.03, 0.01)
    with t3:
        x_thr = st.slider("X cancel threshold", 0.10, 0.90, 0.55, 0.05)
    with t4:
        id_blank = st.slider("ID blank threshold", 0.02, 0.35, 0.10, 0.01)

    st.markdown("---")

    # --- Train button
    if st.button("🚀 تدريب/إعادة تدريب من الأنسر", type="primary"):
        if not key_file:
            st.error("❌ ارفع Answer Key أولاً")
        else:
            try:
                st.session_state.last_error = ""
                pages = load_pages(read_bytes(key_file), key_file.name, dpi=int(dpi))
                key_bgr = pil_to_bgr(pages[0])

                template, train_dbg = learn_template_from_key(key_bgr)

                # Extract answer key from the same key image (with X-rule too)
                key_answers, key_debug = extract_answers(key_bgr, template, blank_thr, double_gap, x_thr)

                st.session_state.template = template
                st.session_state.train_dbg = train_dbg
                st.session_state.overlay = make_overlay(template)
                st.session_state.answer_key = key_answers
                st.session_state.trained = True

                st.success("✅ التدريب نجح + تم استخراج Answer Key.")
            except Exception as e:
                st.session_state.trained = False
                st.session_state.last_error = traceback.format_exc()
                st.error(f"❌ فشل التدريب: {e}")

    # --- Show training results
    if st.session_state.trained and st.session_state.template is not None:
        t = st.session_state.template
        st.markdown("---")
        a, b, c, d = st.columns(4)
        with a:
            st.metric("عدد الأسئلة المكتشف", int(t.num_q))
        with b:
            st.metric("عدد الخيارات المكتشف", int(t.num_choices))
        with c:
            st.metric("خانات كود الطالب", 4)
        with d:
            st.metric("صفوف كود الطالب", 10)

        st.caption(f"Debug: {st.session_state.train_dbg}")

        # Overlay
        if debug and st.session_state.overlay is not None:
            st.markdown("### Overlay (أخضر=أسئلة، أحمر=كود، أزرق=أعمدة الخيارات)")
            st.image(bgr_to_rgb(st.session_state.overlay), width="stretch")

        # ✅ Show Answer Key BEFORE grading (requested)
        st.markdown("### 🔑 Answer Key المستخرج (تأكد قبل التصحيح)")
        ak = st.session_state.answer_key or {}
        df_ak = pd.DataFrame(
            [{"Question": q, "Correct": ans} for q, ans in sorted(ak.items(), key=lambda x: x[0])]
        )
        st.dataframe(df_ak, width="stretch", height=320)

        # Download answer key as CSV
        ak_csv = df_ak.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ تحميل Answer Key CSV", ak_csv, "answer_key.csv", mime="text/csv")

        st.markdown("---")

        # ============================================================
        # Roster
        # ============================================================
        st.header("📋 ملف الطلاب (Roster)")
        roster_file = st.file_uploader("ارفع ملف Excel/CSV للطلاب", type=["xlsx", "xls", "csv"])
        roster_map = {}

        if roster_file:
            try:
                if roster_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(roster_file)
                else:
                    df = pd.read_excel(roster_file)

                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                st.write("معاينة ملف الطلاب:")
                st.dataframe(df.head(10), width="stretch")

                code_col = st.selectbox("اختر عمود كود الطالب", df.columns, index=0)
                name_col = st.selectbox("اختر عمود اسم الطالب", df.columns, index=1 if len(df.columns) > 1 else 0)

                roster_map = dict(
                    zip(df[code_col].astype(str).str.strip(), df[name_col].astype(str).str.strip())
                )
                st.success(f"✅ تم تحميل {len(roster_map)} طالب")
            except Exception as e:
                st.error(f"❌ خطأ في قراءة ملف الطلاب: {e}")

        # ============================================================
        # Student sheets
        # ============================================================
        st.header("📝 أوراق الطلاب")
        student_files = st.file_uploader(
            "ارفع أوراق الطلاب (PDF/صور) — يمكن رفع عدة ملفات",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True
        )

        strict = st.checkbox("✓ وضع صارم (BLANK/DOUBLE/CANCELLED تُحسب خطأ)", value=True)

        if st.button("✅ ابدأ التصحيح", type="primary"):
            if not roster_map:
                st.error("❌ ارفع ملف الطلاب أولاً واختر الأعمدة")
            elif not student_files:
                st.error("❌ ارفع أوراق الطلاب")
            else:
                results_rows = []
                with st.spinner("⏳ جاري التصحيح..."):
                    for f in student_files:
                        try:
                            pages = load_pages(read_bytes(f), f.name, dpi=int(dpi))
                            bgr = pil_to_bgr(pages[0])

                            student_id = extract_student_id(bgr, t, blank_thr_id=id_blank)
                            student_name = roster_map.get(student_id, "غير موجود")

                            stu_answers, stu_dbg = extract_answers(bgr, t, blank_thr, double_gap, x_thr)

                            # grade
                            correct = 0
                            total = len(ak)
                            for q, corr in ak.items():
                                got = stu_answers.get(q, "?")
                                if strict:
                                    # strict: only OK counts, else wrong
                                    st_status = stu_dbg[q]["status"]
                                    if st_status != "OK":
                                        continue
                                if got == corr:
                                    correct += 1

                            perc = (correct / total * 100.0) if total > 0 else 0.0

                            results_rows.append({
                                "file": f.name,
                                "student_id": student_id,
                                "student_name": student_name,
                                "score": correct,
                                "total": total,
                                "percentage": round(perc, 2),
                                "passed": "✓" if perc >= 50 else "✗"
                            })

                        except Exception as e:
                            results_rows.append({
                                "file": f.name,
                                "student_id": "ERROR",
                                "student_name": str(e),
                                "score": 0,
                                "total": len(ak),
                                "percentage": 0,
                                "passed": "✗"
                            })

                df_res = pd.DataFrame(results_rows)
                st.success("✅ اكتمل التصحيح")
                st.dataframe(df_res, width="stretch", height=280)

                # Export Excel
                buf = io.BytesIO()
                df_res.to_excel(buf, index=False)
                st.download_button(
                    "⬇️ تحميل النتائج Excel",
                    buf.getvalue(),
                    "results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # error details
    if st.session_state.last_error:
        st.markdown("---")
        st.error("حدث خطأ. افتح التفاصيل بالأسفل.")
        with st.expander("تفاصيل الخطأ (Traceback)"):
            st.code(st.session_state.last_error)


if __name__ == "__main__":
    main()
