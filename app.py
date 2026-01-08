# app.py
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
# Utils
# =========================
def safe_float(x) -> float:
    if isinstance(x, np.ndarray):
        x = x.ravel()
        return float(x[0]) if x.size else 0.0
    try:
        return float(x)
    except Exception:
        return 0.0


def safe_int(x) -> int:
    return int(round(safe_float(x)))


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
# Bubble detection (STRONGER FILTER)
# =========================
def find_bubble_centers(bin_img: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((0, 2), dtype=np.int32)

    records = []
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

        records.append((area, circ, ar, c))

    if not records:
        return np.zeros((0, 2), dtype=np.int32)

    # learn typical size window
    areas = np.array([r[0] for r in records], dtype=np.float32)
    loga = np.log(np.clip(areas, 1, None))
    hist, edges = np.histogram(loga, bins=25)
    peak = int(np.argmax(hist))
    lo = float(edges[peak])
    hi = float(edges[min(peak + 1, len(edges) - 1)])

    area_lo = float(np.exp(lo)) * 0.45
    area_hi = float(np.exp(hi)) * 3.0

    centers = []
    for area, circ, ar, c in records:
        # IMPORTANT: stricter to avoid text/number/X contours
        if not (area_lo <= area <= area_hi):
            continue
        if circ < 0.35:
            continue
        if not (0.75 <= ar <= 1.35):
            continue

        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6:
            continue
        cx = safe_int(M["m10"] / M["m00"])
        cy = safe_int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    return np.array(centers, dtype=np.int32) if centers else np.zeros((0, 2), dtype=np.int32)


# =========================
# Clustering (no sklearn)
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
# Smart region split (validated)
# =========================
def split_regions(centers: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    if centers.shape[0] < 40:
        return centers, centers

    xs = centers[:, 0].astype(np.float32)
    x_splits = np.percentile(xs, [35, 40, 45, 50, 55, 60, 65])

    def score_id(block: np.ndarray) -> float:
        if block.shape[0] < 30:
            return -1e9
        ry = group_1d_positions(block[:, 1])
        cx = group_1d_positions(block[:, 0])
        return -abs(len(ry) - 10) * 70 - abs(len(cx) - 4) * 120 + block.shape[0] * 0.2

    def score_q(block: np.ndarray) -> float:
        if block.shape[0] < 25:
            return -1e9
        ry = group_1d_positions(block[:, 1])
        cx = group_1d_positions(block[:, 0])
        bestc = min([abs(len(cx) - k) for k in (2, 4, 5)]) if len(cx) else 99
        return len(ry) * 10 - bestc * 60 + block.shape[0] * 0.05

    best = None
    best_score = -1e18

    for s in x_splits:
        left = centers[centers[:, 0] <= s]
        right = centers[centers[:, 0] > s]

        sc1 = score_id(right) + score_q(left)   # right=id, left=q
        sc2 = score_id(left) + score_q(right)   # left=id, right=q

        if sc1 > best_score:
            best_score = sc1
            best = (right, left)
        if sc2 > best_score:
            best_score = sc2
            best = (left, right)

    id_c, q_c = best if best is not None else (centers, centers)
    return id_c, q_c


# =========================
# Strong column picking by FREQUENCY (prevents question-number column)
# =========================
def top_x_peaks(q_centers: np.ndarray, k: int) -> np.ndarray:
    xs = q_centers[:, 0].astype(np.float32)
    x_clusters = group_1d_positions(xs)
    if len(x_clusters) == 0:
        return np.array([], dtype=np.float32)

    # adaptive tol_x
    xs_sort = np.sort(xs)
    dx = np.diff(xs_sort)
    dx = dx[(dx > 2) & (dx < 200)]
    tol_x = max(10.0, float(np.percentile(dx, 15)) * 0.60) if len(dx) else 14.0

    counts = []
    for x0 in x_clusters:
        cnt = int(np.sum(np.abs(xs - x0) < tol_x))
        counts.append((cnt, x0))

    counts.sort(reverse=True, key=lambda t: t[0])
    picked = [x for _, x in counts[:k]]
    return np.sort(np.array(picked, dtype=np.float32))


def build_question_rows(q_centers: np.ndarray, col_x: np.ndarray) -> np.ndarray:
    if q_centers.shape[0] < 10 or len(col_x) < 2:
        return np.array([], dtype=np.float32)

    row_y_all = group_1d_positions(q_centers[:, 1])
    if len(row_y_all) == 0:
        return row_y_all

    xs = q_centers[:, 0].astype(np.float32)
    ys = q_centers[:, 1].astype(np.float32)

    xs_sort = np.sort(xs)
    ys_sort = np.sort(ys)

    dx = np.diff(xs_sort); dx = dx[(dx > 2) & (dx < 250)]
    dy = np.diff(ys_sort); dy = dy[(dy > 2) & (dy < 250)]

    tol_x = max(10.0, float(np.percentile(dx, 15)) * 0.60) if len(dx) else 14.0
    tol_y = max(10.0, float(np.percentile(dy, 15)) * 0.60) if len(dy) else 12.0

    k = len(col_x)
    need = max(2, k - 1)  # not strict "k"

    valid_rows = []
    for y0 in row_y_all:
        near = q_centers[np.abs(q_centers[:, 1] - y0) < tol_y]
        if near.shape[0] < need:
            continue

        hits = 0
        for x0 in col_x:
            if np.any(np.abs(near[:, 0] - x0) < tol_x):
                hits += 1

        if hits >= need:
            valid_rows.append(y0)

    return np.array(valid_rows, dtype=np.float32)


# =========================
# Alignment (ORB)
# =========================
def orb_align(student_bgr: np.ndarray, ref_bgr: np.ndarray) -> Tuple[np.ndarray, bool, int]:
    h, w = ref_bgr.shape[:2]
    orb = cv2.ORB_create(5000)

    g1 = cv2.cvtColor(student_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)

    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        return cv2.resize(student_bgr, (w, h)), False, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 30:
        return cv2.resize(student_bgr, (w, h)), False, len(good)

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return cv2.resize(student_bgr, (w, h)), False, len(good)

    warped = cv2.warpPerspective(student_bgr, H, (w, h))
    return warped, True, len(good)


# =========================
# Bubble scoring (Fill + X)
# =========================
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
    return float(np.mean(th > 0))


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


# =========================
# Smart pick: most shaded wins + ignore X cancelled
# + baseline removal (letters A/B/C/D)
# =========================
def pick_choice_smart(stats_list: List[dict], labels: List[str],
                      blank_norm: float,
                      close_ratio: float,
                      x_std: float,
                      x_score: float) -> Tuple[str, str, List[bool], List[float]]:

    fills = np.array([float(s["fill"]) for s in stats_list], dtype=np.float32)
    stds  = np.array([float(s["std"]) for s in stats_list], dtype=np.float32)
    xs    = np.array([float(s["xscore"]) for s in stats_list], dtype=np.float32)

    cancelled = (stds >= x_std) & (xs >= x_score)

    sorted_f = np.sort(fills)
    k = max(1, len(sorted_f) // 2)
    baseline = float(np.median(sorted_f[:k]))
    norm = np.clip(fills - baseline, 0.0, 1.0)

    cand = [i for i in range(len(labels)) if not cancelled[i]]
    if not cand:
        return "?", "CANCELLED_ALL", cancelled.tolist(), norm.tolist()

    cand.sort(key=lambda i: norm[i], reverse=True)
    best = cand[0]
    best_v = float(norm[best])
    second_v = float(norm[cand[1]]) if len(cand) > 1 else 0.0

    if best_v < blank_norm:
        return "?", "BLANK", cancelled.tolist(), norm.tolist()

    # if close, still pick the best (your rule)
    if second_v > blank_norm and (best_v / (second_v + 1e-9)) < close_ratio:
        return labels[best], "OK_CLOSE", cancelled.tolist(), norm.tolist()

    return labels[best], "OK", cancelled.tolist(), norm.tolist()


# =========================
# Roster
# =========================
def load_roster(file) -> Dict[str, str]:
    if file.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "student_code" not in df.columns or "student_name" not in df.columns:
        raise ValueError("ملف الطلاب لازم يحتوي: student_code و student_name")

    codes = (
        df["student_code"].astype(str).str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(4)
    )
    names = df["student_name"].astype(str).str.strip()
    return dict(zip(codes, names))


# =========================
# Learn template from Answer Key (AUTO REAL)
# =========================
def learn_template_from_key(key_bgr: np.ndarray) -> Tuple[LearnedTemplate, dict]:
    h, w = key_bgr.shape[:2]
    bin_img = preprocess_binary(key_bgr)
    centers = find_bubble_centers(bin_img)

    if centers.shape[0] < 40:
        raise ValueError("عدد الفقاعات المكتشفة قليل جداً. ارفع DPI أو تأكد وضوح الصورة.")

    id_centers, q_centers = split_regions(centers, w, h)
    if id_centers.shape[0] < 30:
        raise ValueError("فشل اكتشاف منطقة كود الطالب تلقائياً.")
    if q_centers.shape[0] < 25:
        raise ValueError("فشل اكتشاف منطقة الأسئلة تلقائياً.")

    # ---- ID grid: force 10 rows + choose best 4 cols
    id_row_y = group_1d_positions(id_centers[:, 1])
    id_col_x = group_1d_positions(id_centers[:, 0])

    # merge/synthesize to 10 rows
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

    id_grid_centers = snap_to_grid(id_centers, id_row_y, id_col_x)
    id_grid = BubbleGrid(id_row_y, id_col_x, id_grid_centers, rows=10, cols=4)

    # ---- Question columns: choose among 2/4/5 using frequency peaks
    best_cols = None
    best_score = -1e18

    for k in (2, 4, 5):
        cols_k = top_x_peaks(q_centers, k)
        if len(cols_k) != k:
            continue
        # score by total hits around these cols
        score = 0
        for x0 in cols_k:
            score += int(np.sum(np.abs(q_centers[:, 0] - x0) < 14))
        if score > best_score:
            best_score = score
            best_cols = cols_k

    if best_cols is None:
        raise ValueError("لم أستطع تحديد عدد الخيارات تلقائياً (2/4/5).")

    q_col_x = best_cols
    num_choices = int(len(q_col_x))

    # ---- Question rows: real rows only (based on hitting columns)
    q_row_y = build_question_rows(q_centers, q_col_x)
    if len(q_row_y) < 3:
        raise ValueError("لم يتم اكتشاف صفوف أسئلة كافية. ربما الصفحة مقصوصة أو غير واضحة.")

    q_grid_centers = snap_to_grid(q_centers, q_row_y, q_col_x)
    q_grid = BubbleGrid(q_row_y, q_col_x, q_grid_centers, rows=len(q_row_y), cols=num_choices)

    template = LearnedTemplate(
        ref_bgr=key_bgr, ref_w=w, ref_h=h,
        id_grid=id_grid,
        q_grid=q_grid,
        num_q=len(q_row_y),
        num_choices=num_choices,
        id_rows=10, id_digits=4
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
# Extract Answer Key
# =========================
def extract_answer_key(template: LearnedTemplate) -> Tuple[Dict[int, str], pd.DataFrame, dict]:
    gray = cv2.cvtColor(template.ref_bgr, cv2.COLOR_BGR2GRAY)
    choices = list("ABCDEFGH"[:template.num_choices])

    blank_norm = 0.06
    close_ratio = 1.12
    x_std_key = 18.0
    x_score_key = 0.90

    key = {}
    dbg_rows = []

    for r in range(template.q_grid.rows):
        stats = []
        for c in range(template.q_grid.cols):
            cx, cy = template.q_grid.centers[r, c]
            stats.append(bubble_stats(gray, safe_int(cx), safe_int(cy), win=18))

        ans, status, cancelled, normfills = pick_choice_smart(
            stats, choices,
            blank_norm=blank_norm,
            close_ratio=close_ratio,
            x_std=x_std_key,
            x_score=x_score_key
        )

        if status.startswith("OK"):
            key[r + 1] = ans

        dbg_rows.append(
            [r + 1, ans, status, cancelled] +
            [round(float(x), 3) for x in normfills] +
            [round(float(s["fill"]), 3) for s in stats] +
            [round(float(s["std"]), 1) for s in stats] +
            [round(float(s["xscore"]), 2) for s in stats]
        )

    df_dbg = pd.DataFrame(
        dbg_rows,
        columns=(["Q", "Picked", "Status", "CancelledFlags"] +
                 [f"norm_{c}" for c in choices] +
                 [f"fill_{c}" for c in choices] +
                 [f"std_{c}" for c in choices] +
                 [f"x_{c}" for c in choices])
    )

    params = {
        "choices": choices,
        "blank_norm": blank_norm,
        "close_ratio": close_ratio,
        "x_std_student": 22.0,
        "x_score_student": 1.2,
        "x_std_key": x_std_key,
        "x_score_key": x_score_key,
    }
    return key, df_dbg, params


# =========================
# Read student ID (10x4)
# =========================
def read_student_id(template: LearnedTemplate, gray: np.ndarray) -> str:
    digits = []
    for c in range(4):
        fills = []
        for r in range(10):
            cx, cy = template.id_grid.centers[r, c]
            roi = bubble_roi(gray, safe_int(cx), safe_int(cy), win=16)
            fills.append(fill_score(roi))

        idx = np.argsort(fills)[::-1]
        best = int(idx[0])
        second = int(idx[1]) if len(idx) > 1 else best

        best_f = float(fills[best])
        second_f = float(fills[second])

        # conservative
        if best_f < 0.08 or (best_f - second_f) < 0.04:
            digit = "X"
        else:
            digit = str(best)  # row index = digit

        digits.append(digit)

    return "".join(digits)


# =========================
# Read student answers
# =========================
def read_student_answers(template: LearnedTemplate, gray: np.ndarray, params: dict) -> pd.DataFrame:
    choices = params["choices"]
    rows = []
    for r in range(template.q_grid.rows):
        stats = []
        for c in range(template.q_grid.cols):
            cx, cy = template.q_grid.centers[r, c]
            stats.append(bubble_stats(gray, safe_int(cx), safe_int(cy), win=18))

        ans, status, cancelled, normfills = pick_choice_smart(
            stats, choices,
            blank_norm=float(params["blank_norm"]),
            close_ratio=float(params["close_ratio"]),
            x_std=float(params["x_std_student"]),
            x_score=float(params["x_score_student"])
        )

        rows.append([r + 1, ans, status, cancelled] + [round(float(x), 3) for x in normfills])

    return pd.DataFrame(rows, columns=["Q", "Picked", "Status", "CancelledFlags"] + [f"norm_{c}" for c in choices])


# =========================
# UI
# =========================
def main():
    st.set_page_config(page_title="AUTO OMR (Robust)", layout="wide")
    st.title("✅ OMR ذكي: اكتشاف الأسئلة + عدد الخيارات (2/4/5) + قاعدة X (إهمال الملغي)")

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
        st.info("ارفع Answer Key أولاً.")
        return

    if (not st.session_state.trained) or st.button("🔄 إعادة التدريب من الأنسر"):
        try:
            key_pages = load_pages(read_bytes(key_file), key_file.name, dpi=int(dpi))
            key_bgr = pil_to_bgr(key_pages[0])

            template, train_dbg = learn_template_from_key(key_bgr)
            answer_key, df_key_dbg, params = extract_answer_key(template)

            # overlay preview
            vis = key_bgr.copy()

            # draw question grid points
            for r in range(template.q_grid.rows):
                for c in range(template.q_grid.cols):
                    x, y = template.q_grid.centers[r, c]
                    cv2.circle(vis, (safe_int(x), safe_int(y)), 6, (0, 255, 0), 2)

            # draw ID grid points
            for r in range(template.id_grid.rows):
                for c in range(template.id_grid.cols):
                    x, y = template.id_grid.centers[r, c]
                    cv2.circle(vis, (safe_int(x), safe_int(y)), 6, (0, 0, 255), 2)

            # draw vertical lines for chosen choice columns (helps verify)
            for x0 in template.q_grid.col_x:
                cv2.line(vis, (safe_int(x0), 0), (safe_int(x0), template.ref_h), (255, 0, 0), 2)

            st.session_state.template = template
            st.session_state.answer_key = answer_key
            st.session_state.key_dbg = df_key_dbg
            st.session_state.params = params
            st.session_state.preview_img = vis
            st.session_state.train_dbg = train_dbg
            st.session_state.trained = True

            st.success("✅ تم التدريب بنجاح.")
        except Exception as e:
            st.error(f"❌ فشل التدريب التلقائي: {e}")
            with st.expander("تفاصيل الخطأ (Traceback)"):
                st.code(traceback.format_exc())
            st.session_state.trained = False
            return

    template = st.session_state.template
    answer_key = st.session_state.answer_key
    df_key_dbg = st.session_state.key_dbg
    params = st.session_state.params

    st.markdown("---")
    st.subheader("📌 نتيجة الاكتشاف قبل التصحيح")

    cA, cB, cC, cD = st.columns(4)
    with cA:
        st.metric("عدد الأسئلة المكتشف", int(template.num_q))
    with cB:
        st.metric("عدد الخيارات المكتشف", int(template.num_choices))
    with cC:
        st.metric("خانات كود الطالب", 4)
    with cD:
        st.metric("صفوف كود الطالب", 10)

    st.caption(f"Debug: {st.session_state.train_dbg}")

    st.markdown("### 🔑 Answer Key المستخرج")
    st.json(answer_key)

    if debug:
        st.markdown("### 🖼️ Overlay للتحقق (أخضر=أسئلة، أحمر=كود، أزرق=أعمدة الخيارات)")
        st.image(bgr_to_rgb(st.session_state.preview_img), width="stretch")

        st.markdown("### جدول Debug (مهم لمعرفة أين يحدث BLANK أو OK_CLOSE)")
        st.dataframe(df_key_dbg, width="stretch", height=450)

    st.markdown("---")
    st.subheader("✅ التصحيح")

    if not roster_file or not sheets_file:
        st.warning("ارفع ملف الطلاب + أوراق الطلاب للبدء.")
        return

    try:
        roster = load_roster(roster_file)
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

            results = []
            prog = st.progress(0)

            for i, pil_page in enumerate(pages, start=1):
                page_bgr = pil_to_bgr(pil_page)
                aligned, ok, good = orb_align(page_bgr, template.ref_bgr)
                gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

                code = read_student_id(template, gray)
                code = str(code).zfill(4)
                name = roster.get(code, "غير موجود")

                df_ans = read_student_answers(template, gray, params)

                correct = 0
                total = len(answer_key)

                for _, row in df_ans.iterrows():
                    q = int(row["Q"])
                    if q not in answer_key:
                        continue
                    if not str(row["Status"]).startswith("OK"):
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
            with st.expander("تفاصيل الخطأ (Traceback)"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
