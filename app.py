# app.py
# ======================================================================================
# Robust Hybrid OMR + Smart Cancel-X
# Fix: Prevent swapping ID grid and Question grid by enforcing "right-most ID grid"
# ======================================================================================

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# pdf2image (PDF support)
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_OK = True
except Exception:
    PDF2IMAGE_OK = False


# ======================================================================================
# Data models
# ======================================================================================

@dataclass
class GridSpec:
    centers: List[Tuple[float, float]]  # (x,y) of detected bubbles in this grid region
    col_x: List[float]                  # representative column x positions
    row_y: List[float]                  # representative row y positions
    n_rows: int
    n_cols: int
    bbox: Tuple[int, int, int, int]     # (x1,y1,x2,y2) bounding box of grid


@dataclass
class LearnedTemplate:
    page_w: int
    page_h: int
    id_grid: GridSpec                   # MUST be 10 rows × 4 cols
    q_grid: GridSpec                    # Q rows × choices cols
    num_choices: int                    # 2/4/5
    num_questions: int                  # Q
    bubble_r: int


# ======================================================================================
# Basic I/O + image helpers
# ======================================================================================

def load_first_page(file_bytes: bytes, filename: str, dpi: int = 300) -> Image.Image:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        if not PDF2IMAGE_OK:
            raise RuntimeError("pdf2image غير مثبت. ثبّت pdf2image + poppler في البيئة.")
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        if not pages:
            raise RuntimeError("لم يتم استخراج صفحات من PDF.")
        return pages[0].convert("RGB")
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def preprocess_for_bubbles(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 8
    )
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return th


# ======================================================================================
# Bubble detection
# ======================================================================================

def find_circle_like_bubbles(binary: np.ndarray,
                             min_area: int = 120,
                             max_area: int = 9000,
                             min_circularity: float = 0.30):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        per = cv2.arcLength(c, True)
        if per <= 1e-6:
            continue
        circ = 4.0 * math.pi * area / (per * per)
        if circ < min_circularity:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 3 or h <= 3:
            continue
        ratio = w / float(h)
        if ratio < 0.55 or ratio > 1.8:
            continue
        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        bubbles.append((cx, cy, area, (x, y, w, h)))
    return bubbles


def robust_unique_centers(points: List[Tuple[float, float]],
                          x_eps: float = 8.0,
                          y_eps: float = 8.0) -> List[Tuple[float, float]]:
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    out = []
    for (x, y) in pts:
        ok = True
        for (ox, oy) in out[-40:]:
            if abs(x - ox) <= x_eps and abs(y - oy) <= y_eps:
                ok = False
                break
        if ok:
            out.append((x, y))
    return out


def cluster_1d(values: np.ndarray, eps: float) -> List[np.ndarray]:
    if len(values) == 0:
        return []
    v = np.sort(values)
    clusters = []
    cur = [v[0]]
    for i in range(1, len(v)):
        if v[i] - v[i - 1] <= eps:
            cur.append(v[i])
        else:
            clusters.append(np.array(cur))
            cur = [v[i]]
    clusters.append(np.array(cur))
    return clusters


def estimate_spacing(values_sorted: np.ndarray) -> float:
    if len(values_sorted) < 3:
        return 30.0
    diffs = np.diff(values_sorted)
    diffs = diffs[diffs > 1.0]
    if len(diffs) == 0:
        return 30.0
    return float(np.median(diffs))


def grid_bbox_from_cols_rows(col_x: List[float], row_y: List[float], pad: int = 35) -> Tuple[int, int, int, int]:
    x1 = int(min(col_x) - pad)
    x2 = int(max(col_x) + pad)
    y1 = int(min(row_y) - pad)
    y2 = int(max(row_y) + pad)
    return (x1, y1, x2, y2)


def build_grid_from_centers(centers: List[Tuple[float, float]],
                            n_cols: int,
                            y_rows_expected: Optional[int] = None) -> Optional[GridSpec]:
    if len(centers) < n_cols * 6:
        return None

    xs = np.array([c[0] for c in centers], dtype=np.float32)
    ys = np.array([c[1] for c in centers], dtype=np.float32)

    xs_sorted = np.sort(xs)
    x_spacing = estimate_spacing(xs_sorted)
    x_clusters = cluster_1d(xs, eps=max(10.0, x_spacing * 0.55))
    x_means = np.array([np.mean(cl) for cl in x_clusters], dtype=np.float32)
    x_counts = np.array([len(cl) for cl in x_clusters], dtype=np.int32)

    if len(x_means) < n_cols:
        return None

    idx = np.argsort(x_means)
    x_means = x_means[idx]
    x_counts = x_counts[idx]

    best = None
    for i in range(0, len(x_means) - n_cols + 1):
        window = x_means[i:i + n_cols]
        w_counts = x_counts[i:i + n_cols]
        dif = np.diff(window)
        spacing_irreg = float(np.std(dif) / (np.mean(dif) + 1e-6))
        score = float(np.sum(w_counts)) - 40.0 * spacing_irreg
        if best is None or score > best[0]:
            best = (score, window)

    if best is None:
        return None

    col_x = sorted([float(x) for x in best[1]])

    # keep centers close to chosen columns
    col_arr = np.array(col_x, dtype=np.float32)
    assigned = []
    for (x, y) in centers:
        j = int(np.argmin(np.abs(col_arr - x)))
        if abs(col_arr[j] - x) <= max(18.0, x_spacing * 0.70):
            assigned.append((x, y))
    if len(assigned) < n_cols * 6:
        return None

    ys2 = np.array([p[1] for p in assigned], dtype=np.float32)
    ys_sorted = np.sort(ys2)
    y_spacing = estimate_spacing(ys_sorted)
    y_clusters = cluster_1d(ys2, eps=max(10.0, y_spacing * 0.55))
    row_y = sorted([float(np.mean(cl)) for cl in y_clusters])

    if y_rows_expected is not None:
        if len(row_y) < y_rows_expected:
            return None
        # reduce/merge to expected if more
        while len(row_y) > y_rows_expected:
            dif = [row_y[k + 1] - row_y[k] for k in range(len(row_y) - 1)]
            kmin = int(np.argmin(dif))
            merged = 0.5 * (row_y[kmin] + row_y[kmin + 1])
            row_y = row_y[:kmin] + [merged] + row_y[kmin + 2:]

    bbox = grid_bbox_from_cols_rows(col_x, row_y, pad=35)
    return GridSpec(
        centers=assigned,
        col_x=col_x,
        row_y=row_y,
        n_rows=len(row_y),
        n_cols=len(col_x),
        bbox=bbox
    )


def points_outside_bbox(points: List[Tuple[float, float]], bbox: Tuple[int, int, int, int]) -> List[Tuple[float, float]]:
    x1, y1, x2, y2 = bbox
    out = []
    for (x, y) in points:
        if x1 <= x <= x2 and y1 <= y <= y2:
            continue
        out.append((x, y))
    return out


# ======================================================================================
# Fill + Cancel X scoring
# ======================================================================================

def crop_around(binary: np.ndarray, cx: float, cy: float, r: int) -> np.ndarray:
    h, w = binary.shape[:2]
    x1 = max(0, int(cx - r))
    x2 = min(w, int(cx + r))
    y1 = max(0, int(cy - r))
    y2 = min(h, int(cy + r))
    return binary[y1:y2, x1:x2].copy()


def fill_score(binary: np.ndarray, cx: float, cy: float, r: int) -> float:
    roi = crop_around(binary, cx, cy, r)
    if roi.size == 0:
        return 0.0
    hh, ww = roi.shape[:2]
    m = int(min(hh, ww) * 0.22)
    inner = roi[m:hh - m, m:ww - m] if (hh - 2 * m > 3 and ww - 2 * m > 3) else roi
    return float(np.mean(inner > 0))


def x_cancel_score(binary: np.ndarray, cx: float, cy: float, r: int) -> float:
    roi = crop_around(binary, cx, cy, r)
    if roi.size == 0:
        return 0.0
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                            minLineLength=max(10, r // 2),
                            maxLineGap=6)
    if lines is None:
        return 0.0
    diag = 0
    total = 0
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, l)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < 2 and dy < 2:
            continue
        total += 1
        angle = math.degrees(math.atan2(dy, dx + 1e-6))
        if 25 <= angle <= 75:
            diag += 1
    if total == 0:
        return 0.0
    density = float(np.mean(edges > 0))
    return float(min(1.0, (diag / float(total)) * (density * 8.0 + 0.2)))


def choose_answer(binary: np.ndarray,
                  centers: List[Tuple[float, float]],
                  labels: List[str],
                  r: int,
                  blank_thr: float,
                  double_gap: float,
                  x_thr: float) -> Dict:
    fills = np.array([fill_score(binary, x, y, r) for (x, y) in centers], dtype=np.float32)
    xsc = np.array([x_cancel_score(binary, x, y, r) for (x, y) in centers], dtype=np.float32)
    canceled = xsc >= x_thr

    cand = np.where(~canceled)[0]
    if len(cand) == 0:
        return {"answer": "?", "status": "BLANK", "fills": fills.tolist(), "x": xsc.tolist(), "canceled": canceled.tolist()}

    sub = fills[cand]
    best_local = int(np.argmax(sub))
    best_idx = int(cand[best_local])
    best_fill = float(fills[best_idx])

    if best_fill < blank_thr:
        return {"answer": "?", "status": "BLANK", "fills": fills.tolist(), "x": xsc.tolist(), "canceled": canceled.tolist()}

    sorted_sub = np.sort(sub)[::-1]
    second = float(sorted_sub[1]) if len(sorted_sub) > 1 else 0.0
    if second > blank_thr and (best_fill - second) < double_gap:
        return {"answer": labels[best_idx], "status": "DOUBLE", "fills": fills.tolist(), "x": xsc.tolist(), "canceled": canceled.tolist()}

    return {"answer": labels[best_idx], "status": "OK", "fills": fills.tolist(), "x": xsc.tolist(), "canceled": canceled.tolist()}


# ======================================================================================
# Template learning (FIXED SWAP PROBLEM HERE)
# ======================================================================================

def learn_template_from_key(key_bgr: np.ndarray,
                            min_area: int,
                            max_area: int,
                            min_circularity: float) -> Tuple[LearnedTemplate, Dict]:
    H, W = key_bgr.shape[:2]
    binary = preprocess_for_bubbles(key_bgr)
    bubbles = find_circle_like_bubbles(binary, min_area=min_area, max_area=max_area, min_circularity=min_circularity)

    centers_all = robust_unique_centers([(cx, cy) for (cx, cy, _, _) in bubbles], x_eps=8.0, y_eps=8.0)
    if len(centers_all) < 60:
        raise ValueError("عدد الفقاعات المكتشفة قليل جداً. ارفع DPI أو حسّن وضوح الصفحة.")

    # estimate bubble radius
    bbox_ws = [bb[2] for *_, bb in bubbles]
    bbox_hs = [bb[3] for *_, bb in bubbles]
    med = int(np.median(np.array(bbox_ws + bbox_hs))) if len(bbox_ws) else 34
    bubble_r = max(12, int(med * 0.55))

    # --- 1) Find ALL candidate 4x10 grids and pick the RIGHT-MOST one as ID
    id_candidates: List[GridSpec] = []
    # Try on whole set (we will score by right-most + narrow width)
    gs = build_grid_from_centers(centers_all, n_cols=4, y_rows_expected=10)
    if gs is not None and gs.n_rows == 10 and gs.n_cols == 4:
        id_candidates.append(gs)

    # Also try by splitting x into multiple regions: left/mid/right bands
    xs = np.array([p[0] for p in centers_all], dtype=np.float32)
    q1, q2 = np.quantile(xs, [0.33, 0.66])
    band_left = [p for p in centers_all if p[0] <= q1]
    band_mid = [p for p in centers_all if q1 < p[0] <= q2]
    band_right = [p for p in centers_all if p[0] > q2]

    for band in [band_right, band_mid, band_left]:
        cand = build_grid_from_centers(band, n_cols=4, y_rows_expected=10)
        if cand is not None and cand.n_rows == 10 and cand.n_cols == 4:
            id_candidates.append(cand)

    if not id_candidates:
        raise ValueError("لم يتم اكتشاف شبكة كود الطالب (4×10) بشكل واضح.")

    # score ID candidates:
    # - prefer higher mean x (more right)
    # - prefer smaller width span
    # - prefer more detected points
    best_id = None
    best_score = -1e18
    for cand in id_candidates:
        mean_x = float(np.mean(cand.col_x))
        width_span = float(max(cand.col_x) - min(cand.col_x))
        dens = len(cand.centers)
        score = (mean_x * 1000.0) + (dens * 5.0) - (width_span * 20.0)
        if score > best_score:
            best_score = score
            best_id = cand

    id_grid = best_id

    # --- 2) Remove ID region points completely
    remaining = points_outside_bbox(centers_all, id_grid.bbox)

    # --- 3) Detect question grid from remaining points: cols in {5,4,2}
    q_best = None
    q_best_score = -1e18
    q_cols_best = None

    for ncols in [5, 4, 2]:
        cand = build_grid_from_centers(remaining, n_cols=ncols, y_rows_expected=None)
        if cand is None:
            continue
        # must have enough rows
        if cand.n_rows < 6:
            continue

        # score: many points + large vertical span, also prefer being NOT inside ID area
        vspan = float(max(cand.row_y) - min(cand.row_y)) if cand.row_y else 0.0
        dens = len(cand.centers)

        # reject if candidate bbox overlaps strongly with ID bbox
        ix1, iy1, ix2, iy2 = id_grid.bbox
        qx1, qy1, qx2, qy2 = cand.bbox
        inter_w = max(0, min(ix2, qx2) - max(ix1, qx1))
        inter_h = max(0, min(iy2, qy2) - max(iy1, qy1))
        inter_area = inter_w * inter_h
        q_area = max(1, (qx2 - qx1) * (qy2 - qy1))
        overlap = inter_area / float(q_area)

        score = dens * 5.0 + vspan * 0.5 - overlap * 2000.0
        if score > q_best_score:
            q_best_score = score
            q_best = cand
            q_cols_best = ncols

    if q_best is None:
        raise ValueError("فشل اكتشاف شبكة الأسئلة. ارفع DPI أو تأكد أن صفحة الأنسر كاملة وواضحة.")

    q_grid = q_best
    num_questions = int(q_grid.n_rows)
    num_choices = int(q_cols_best)

    tmpl = LearnedTemplate(
        page_w=W, page_h=H,
        id_grid=id_grid,
        q_grid=q_grid,
        num_choices=num_choices,
        num_questions=num_questions,
        bubble_r=bubble_r
    )

    dbg = {
        "centers_total": len(centers_all),
        "id_bbox": id_grid.bbox,
        "q_bbox": q_grid.bbox,
        "id_cols": id_grid.col_x,
        "q_cols": q_grid.col_x,
        "id_rows": id_grid.n_rows,
        "q_rows": q_grid.n_rows,
        "num_choices": num_choices,
        "bubble_r": bubble_r
    }
    return tmpl, dbg


def overlay_template(bgr: np.ndarray, tmpl: LearnedTemplate) -> np.ndarray:
    out = bgr.copy()
    # ID red
    for (x, y) in tmpl.id_grid.centers:
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), 2)
    # Q green
    for (x, y) in tmpl.q_grid.centers:
        cv2.circle(out, (int(x), int(y)), 6, (0, 255, 0), 2)
    # Q columns blue
    for x in tmpl.q_grid.col_x:
        cv2.line(out, (int(x), 0), (int(x), tmpl.page_h - 1), (255, 0, 0), 2)
    # ID bbox
    x1, y1, x2, y2 = tmpl.id_grid.bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Q bbox
    x1, y1, x2, y2 = tmpl.q_grid.bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


# ======================================================================================
# Assign points to grid
# ======================================================================================

def snap_to_grid(centers: List[Tuple[float, float]],
                 col_x: List[float],
                 row_y: List[float]) -> Dict[Tuple[int, int], Tuple[float, float]]:
    col = np.array(col_x, dtype=np.float32)
    row = np.array(row_y, dtype=np.float32)
    grid = {}
    for (x, y) in centers:
        c = int(np.argmin(np.abs(col - x)))
        r = int(np.argmin(np.abs(row - y)))
        key = (r, c)
        if key not in grid:
            grid[key] = (x, y)
        else:
            ox, oy = grid[key]
            if (abs(col[c] - x) + abs(row[r] - y)) < (abs(col[c] - ox) + abs(row[r] - oy)):
                grid[key] = (x, y)
    return grid


def extract_student_id(binary: np.ndarray, tmpl: LearnedTemplate,
                       blank_thr: float, double_gap: float, x_thr: float) -> str:
    grid = snap_to_grid(tmpl.id_grid.centers, tmpl.id_grid.col_x, tmpl.id_grid.row_y)
    digits = []
    for c in range(4):
        row_centers = []
        for r in range(10):
            row_centers.append(grid.get((r, c), (tmpl.id_grid.col_x[c], tmpl.id_grid.row_y[r])))
        res = choose_answer(binary, row_centers, [str(i) for i in range(10)],
                            tmpl.bubble_r, blank_thr, double_gap, x_thr)
        digits.append(res["answer"] if res["status"] == "OK" else "X")
    return "".join(digits)


def extract_answers(binary: np.ndarray, tmpl: LearnedTemplate,
                    blank_thr: float, double_gap: float, x_thr: float) -> Dict[int, Dict]:
    grid = snap_to_grid(tmpl.q_grid.centers, tmpl.q_grid.col_x, tmpl.q_grid.row_y)
    labels = list("ABCDE")[:tmpl.num_choices]
    out = {}
    for r in range(tmpl.num_questions):
        row_centers = []
        for c in range(tmpl.num_choices):
            row_centers.append(grid.get((r, c), (tmpl.q_grid.col_x[c], tmpl.q_grid.row_y[r])))
        out[r + 1] = choose_answer(binary, row_centers, labels,
                                   tmpl.bubble_r, blank_thr, double_gap, x_thr)
    return out


# ======================================================================================
# Grading
# ======================================================================================

def read_roster_file(uploaded) -> Dict[str, str]:
    if uploaded is None:
        return {}
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    code_col = None
    name_col = None
    for c in df.columns:
        if c in ["student_code", "code", "id", "student_id"]:
            code_col = c
        if c in ["student_name", "name", "full_name"]:
            name_col = c
    if code_col is None or name_col is None:
        raise ValueError("ملف الطلاب يجب أن يحتوي أعمدة مثل: student_code و student_name (أو code/name).")
    return dict(zip(df[code_col].astype(str).str.strip(), df[name_col].astype(str).str.strip()))


def grade_one_sheet(sheet_bgr: np.ndarray,
                    tmpl: LearnedTemplate,
                    answer_key: Dict[int, str],
                    roster: Dict[str, str],
                    strict: bool,
                    blank_thr: float,
                    double_gap: float,
                    x_thr: float) -> Dict:
    aligned = cv2.resize(sheet_bgr, (tmpl.page_w, tmpl.page_h), interpolation=cv2.INTER_AREA)
    binary = preprocess_for_bubbles(aligned)

    sid = extract_student_id(binary, tmpl, blank_thr, double_gap, x_thr)
    name = roster.get(str(sid).strip(), "غير موجود")

    stud = extract_answers(binary, tmpl, blank_thr, double_gap, x_thr)

    total = len(answer_key)
    correct = wrong = blanks = doubles = 0

    for q, k in answer_key.items():
        s = stud.get(q, {"status": "BLANK", "answer": "?"})
        if s["status"] == "BLANK":
            blanks += 1
            if strict:
                wrong += 1
            continue
        if s["status"] == "DOUBLE":
            doubles += 1
            if strict:
                wrong += 1
                continue
        if s["answer"] == k:
            correct += 1
        else:
            wrong += 1

    pct = (correct / total * 100.0) if total > 0 else 0.0
    return {
        "id": sid,
        "name": name,
        "score": correct,
        "total": total,
        "wrong": wrong,
        "blank": blanks,
        "double": doubles,
        "percentage": pct,
        "passed": pct >= 50.0
    }


# ======================================================================================
# Streamlit UI
# ======================================================================================

def main():
    st.set_page_config(page_title="Hybrid OMR + Smart", layout="wide")
    st.title("✅ Hybrid OMR + Smart Cancel-X (مصحّح فقاعات ذكي)")
    st.caption("إصلاح الخلط: كود الطالب = 4×10 ويجب أن يكون أقصى يمين الصفحة (Right-most)")

    c1, c2, c3 = st.columns(3)
    with c1:
        roster_file = st.file_uploader("📋 ملف الطلاب (Excel/CSV)", type=["xlsx", "xls", "csv"])
    with c2:
        key_file = st.file_uploader("🔑 Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    with c3:
        sheet_file = st.file_uploader("🧾 ورقة الطالب (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])

    st.divider()

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        dpi = st.selectbox("DPI للـ PDF", [200, 250, 300, 350], index=2)
    with s2:
        strict = st.checkbox("وضع صارم (BLANK/DOUBLE = خطأ)", value=True)
    with s3:
        debug = st.checkbox("Debug", value=True)
    with s4:
        st.write("")

    st.subheader("Thresholds (Fill + X)")
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        blank_thr = st.slider("Blank fill threshold", 0.05, 0.40, 0.14, 0.01)
    with t2:
        double_gap = st.slider("Double gap threshold", 0.01, 0.20, 0.03, 0.01)
    with t3:
        x_thr = st.slider("X score threshold", 0.20, 1.00, 0.90, 0.05)
    with t4:
        min_area = st.number_input("min_area", 50, 5000, 120, 10)
    with t5:
        min_circ = st.slider("min_circularity", 0.10, 0.95, 0.30, 0.01)

    max_area = 9000

    st.divider()

    if "tmpl" not in st.session_state:
        st.session_state.tmpl = None
    if "answer_key" not in st.session_state:
        st.session_state.answer_key = None
    if "overlay" not in st.session_state:
        st.session_state.overlay = None
    if "dbg" not in st.session_state:
        st.session_state.dbg = None

    if st.button("🎯 تدريب/إعادة تدريب من Answer Key", type="primary", width="stretch"):
        if key_file is None:
            st.error("ارفع ملف Answer Key أولاً.")
        else:
            try:
                key_pil = load_first_page(key_file.getvalue(), key_file.name, dpi=int(dpi))
                key_bgr = pil_to_bgr(key_pil)

                tmpl, dbg = learn_template_from_key(
                    key_bgr,
                    min_area=int(min_area),
                    max_area=int(max_area),
                    min_circularity=float(min_circ),
                )

                # Extract key answers from key itself (to show to user)
                key_aligned = cv2.resize(key_bgr, (tmpl.page_w, tmpl.page_h), interpolation=cv2.INTER_AREA)
                key_bin = preprocess_for_bubbles(key_aligned)
                key_ans_detail = extract_answers(key_bin, tmpl, blank_thr, double_gap, x_thr)

                answer_key = {}
                for q, res in key_ans_detail.items():
                    answer_key[q] = res["answer"] if res["answer"] in list("ABCDE") else "?"

                st.session_state.tmpl = tmpl
                st.session_state.answer_key = answer_key
                st.session_state.dbg = dbg
                st.session_state.overlay = bgr_to_pil(overlay_template(key_aligned, tmpl))

                st.success("✅ التدريب نجح (بدون خلط ID/Questions).")
            except Exception as e:
                st.error(f"❌ فشل التدريب: {e}")

    # Show training output
    if st.session_state.tmpl is not None:
        tmpl = st.session_state.tmpl
        answer_key = st.session_state.answer_key or {}

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("عدد الأسئلة المكتشف", tmpl.num_questions)
        a2.metric("عدد الخيارات المكتشف", tmpl.num_choices)
        a3.metric("خانات كود الطالب", 4)
        a4.metric("صفوف كود الطالب", 10)

        st.subheader("🧠 Answer Key المستخرج (تأكد قبل التصحيح)")
        st.json({str(k): v for k, v in answer_key.items()})

        if st.session_state.overlay is not None:
            st.subheader("Overlay (أحمر=ID, أخضر=Questions, أزرق=أعمدة الخيارات)")
            st.image(st.session_state.overlay, width="stretch")

        if debug and st.session_state.dbg:
            st.caption(f"Debug: {st.session_state.dbg}")

    st.divider()
    st.subheader("✅ التصحيح")

    if st.button("🚀 ابدأ التصحيح الآن", type="primary", width="stretch"):
        if st.session_state.tmpl is None or st.session_state.answer_key is None:
            st.error("لازم تعمل تدريب من Answer Key أولاً.")
        elif roster_file is None:
            st.error("ارفع ملف الطلاب أولاً.")
        elif sheet_file is None:
            st.error("ارفع ورقة الطالب أولاً.")
        else:
            try:
                roster = read_roster_file(roster_file)
                tmpl = st.session_state.tmpl
                answer_key = st.session_state.answer_key

                sheet_pil = load_first_page(sheet_file.getvalue(), sheet_file.name, dpi=int(dpi))
                sheet_bgr = pil_to_bgr(sheet_pil)

                res = grade_one_sheet(
                    sheet_bgr, tmpl, answer_key, roster,
                    strict=bool(strict),
                    blank_thr=float(blank_thr),
                    double_gap=float(double_gap),
                    x_thr=float(x_thr),
                )

                df = pd.DataFrame([{
                    "كود الطالب": res["id"],
                    "اسم الطالب": res["name"],
                    "الإجابات الصحيحة": res["score"],
                    "إجمالي الأسئلة": res["total"],
                    "خاطئة": res["wrong"],
                    "فارغة": res["blank"],
                    "مزدوجة": res["double"],
                    "النسبة": f"{res['percentage']:.1f}%",
                    "الحالة": "ناجح ✓" if res["passed"] else "راسب ✗",
                }])

                st.success("✅ اكتمل التصحيح.")
                st.dataframe(df, width="stretch")

                buf = io.BytesIO()
                df.to_excel(buf, index=False, engine="openpyxl")
                st.download_button(
                    "⬇️ تحميل النتائج (Excel)",
                    data=buf.getvalue(),
                    file_name="results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )
            except Exception as e:
                st.error(f"❌ خطأ أثناء التصحيح: {e}")


if __name__ == "__main__":
    main()
