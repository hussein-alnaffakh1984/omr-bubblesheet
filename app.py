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


# =========================
# Safety / Helpers
# =========================
def _read_bytes(uploaded_file) -> bytes:
    if uploaded_file is None:
        return b""
    try:
        return uploaded_file.getbuffer().tobytes()
    except Exception:
        try:
            return uploaded_file.read()
        except Exception:
            return b""


def _load_pages(file_bytes: bytes, filename: str, dpi: int = 300) -> List[Image.Image]:
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _safe_int(x) -> int:
    try:
        if isinstance(x, np.ndarray):
            x = x.ravel()
            return int(round(float(x[0]))) if x.size else 0
        return int(round(float(x)))
    except Exception:
        return 0


# =========================
# Data models
# =========================
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


# =========================
# Preprocess
# =========================
def preprocess_binary(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )
    bin_img = cv2.medianBlur(bin_img, 3)
    return bin_img


# =========================
# Bubble detection (strong filter to avoid text / numbers)
# =========================
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


# =========================
# Simple clustering (no sklearn)
# =========================
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


# =========================
# Split regions: ID vs Questions
# =========================
def split_regions(centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs = centers[:, 0].astype(np.float32)
    # Try several splits, pick where one side looks like ID (10x4) and other like Q
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


# =========================
# Strong choice columns selection (prevents ID leakage)
# =========================
def top_x_peaks(q_centers: np.ndarray, k: int) -> np.ndarray:
    xs = q_centers[:, 0].astype(np.float32)
    clusters = group_1d_positions(xs)
    if len(clusters) == 0:
        return np.array([], dtype=np.float32)

    # tol based on x diffs
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


def sanitize_q_centers(q_centers: np.ndarray) -> np.ndarray:
    # Keep only middle X band to avoid pulling ID bubbles as question options
    qx = q_centers[:, 0].astype(np.float32)
    x_lo, x_hi = np.percentile(qx, 10), np.percentile(qx, 90)
    return q_centers[(qx >= x_lo) & (qx <= x_hi)]


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
            # reject if one column is far away (typical ID leakage)
            if np.max(dif) > 1.8 * med:
                continue
        # score by hits
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


# =========================
# Learn template (Answer Key)
# =========================
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

    # --- ID grid: force 10 rows and 4 columns
    id_row_y = group_1d_positions(id_centers[:, 1])
    id_col_x = group_1d_positions(id_centers[:, 0])

    # enforce 10 rows
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

    # --- Questions: pick choice columns without ID leakage
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


# =========================
# UI / App
# =========================
def main():
    st.set_page_config(page_title="OMR AUTO", layout="wide")
    st.title("✅ OMR Auto (ثابت + عرض أخطاء واضح)")

    if "trained" not in st.session_state:
        st.session_state.trained = False
        st.session_state.template = None
        st.session_state.train_dbg = None
        st.session_state.preview_img = None
        st.session_state.last_error = ""

    col1, col2 = st.columns(2)
    with col1:
        key_file = st.file_uploader("🔑 Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    with col2:
        dpi = st.selectbox("DPI للـ PDF", [200, 250, 300, 350, 400], index=2)

    debug = st.checkbox("Debug", value=True)

    if not key_file:
        st.info("ارفع Answer Key أولاً.")
        return

    if st.button("🚀 تدريب/إعادة تدريب من الأنسر", type="primary"):
        try:
            st.session_state.last_error = ""
            pages = _load_pages(_read_bytes(key_file), key_file.name, dpi=int(dpi))
            key_bgr = _pil_to_bgr(pages[0])

            template, train_dbg = learn_template_from_key(key_bgr)

            # preview overlay
            vis = key_bgr.copy()

            # question points
            for r in range(template.q_grid.rows):
                for c in range(template.q_grid.cols):
                    x, y = template.q_grid.centers[r, c]
                    cv2.circle(vis, (_safe_int(x), _safe_int(y)), 6, (0, 255, 0), 2)

            # ID points
            for r in range(template.id_grid.rows):
                for c in range(template.id_grid.cols):
                    x, y = template.id_grid.centers[r, c]
                    cv2.circle(vis, (_safe_int(x), _safe_int(y)), 6, (0, 0, 255), 2)

            # choice columns lines
            for x0 in template.q_grid.col_x:
                cv2.line(vis, (_safe_int(x0), 0), (_safe_int(x0), template.ref_h), (255, 0, 0), 2)

            st.session_state.template = template
            st.session_state.train_dbg = train_dbg
            st.session_state.preview_img = vis
            st.session_state.trained = True

            st.success("✅ التدريب نجح.")
        except Exception as e:
            st.session_state.trained = False
            st.session_state.last_error = traceback.format_exc()
            st.error(f"❌ فشل التدريب: {e}")

    if st.session_state.trained and st.session_state.template is not None:
        t = st.session_state.template
        st.markdown("---")
        cA, cB, cC, cD = st.columns(4)
        with cA:
            st.metric("عدد الأسئلة المكتشف", int(t.num_q))
        with cB:
            st.metric("عدد الخيارات المكتشف", int(t.num_choices))
        with cC:
            st.metric("خانات كود الطالب", 4)
        with cD:
            st.metric("صفوف كود الطالب", 10)

        st.caption(f"Debug: {st.session_state.train_dbg}")

        if debug and st.session_state.preview_img is not None:
            st.markdown("### Overlay (أخضر=أسئلة، أحمر=كود، أزرق=أعمدة الخيارات)")
            st.image(_bgr_to_rgb(st.session_state.preview_img), width="stretch")

    if st.session_state.last_error:
        st.markdown("---")
        st.error("حدث خطأ. افتح التفاصيل بالأسفل.")
        with st.expander("تفاصيل الخطأ (Traceback)"):
            st.code(st.session_state.last_error)


if __name__ == "__main__":
    main()
