# app.py
# ============================================================
# Hybrid OMR (Auto-Learn from Answer Key) + Smart X-ignore
# - Auto-detect ID grid (4x10) and Questions grid (N x (2/4/5))
# - Robust alignment (ORB + Homography)
# - Smart rule:
#     If bubble has X mark => ignore it
#     If multiple filled => pick most filled
#     If only X-filled exist => treat as blank/cancelled
# - Shows extracted Answer Key BEFORE grading (for confirmation)
# ============================================================

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes


# -----------------------------
# Utilities
# -----------------------------

def pil_from_upload(file) -> Image.Image:
    data = file.getvalue()
    name = (file.name or "").lower()
    if name.endswith(".pdf"):
        pages = convert_from_bytes(data, dpi=int(st.session_state.get("dpi", 300)))
        if not pages:
            raise ValueError("لم يتم قراءة صفحات PDF.")
        return pages[0].convert("RGB")
    return Image.open(io.BytesIO(data)).convert("RGB")


def pdf_all_pages_bytes(file_bytes: bytes, dpi: int) -> List[Image.Image]:
    pages = convert_from_bytes(file_bytes, dpi=dpi)
    return [p.convert("RGB") for p in pages]


def to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def adaptive_bin_inv(g: np.ndarray) -> np.ndarray:
    g2 = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.adaptiveThreshold(
        g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )


def safe_int(x) -> int:
    try:
        return int(x)
    except:
        return 0


# -----------------------------
# Template model
# -----------------------------

@dataclass
class Grid:
    centers: List[Tuple[float, float]]     # (x, y) list
    cols_x: List[float]                    # sorted x centers for columns
    rows_y: List[float]                    # sorted y centers for rows
    n_cols: int
    n_rows: int


@dataclass
class Template:
    key_size: Tuple[int, int]              # (w, h)
    id_grid: Grid                          # 4 x 10
    q_grid: Grid                           # (2/4/5) x N
    q_choices: int                         # 2 or 4 or 5
    q_count: int                           # N
    key_ref_gray: np.ndarray               # for alignment


# -----------------------------
# Circle detection
# -----------------------------

def find_circle_centers(img_bgr: np.ndarray,
                        min_area: int = 120,
                        max_area: int = 9000,
                        min_circ: float = 0.30) -> List[Tuple[float, float, float]]:
    """
    Returns list of (cx, cy, radius_est) for circular contours.
    Uses contour circularity filter (works better than Hough in many scans).
    """
    g = gray(img_bgr)
    b = adaptive_bin_inv(g)

    # remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 1e-6:
            continue
        circ = (4.0 * math.pi * area) / (peri * peri)
        if circ < min_circ:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        out.append((float(x), float(y), float(r)))
    return out


def cluster_1d(values: List[float], tol: float) -> List[List[float]]:
    """Simple 1D clustering by sorting and grouping if gap <= tol."""
    if not values:
        return []
    v = sorted(values)
    groups = [[v[0]]]
    for x in v[1:]:
        if abs(x - groups[-1][-1]) <= tol:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def pick_grid_from_centers(centers_xy: List[Tuple[float, float]],
                           expected_rows: Optional[int],
                           expected_cols_candidates: List[int],
                           tol_x: float,
                           tol_y: float) -> Tuple[int, int, List[float], List[float], Dict]:
    """
    Decide grid shape by clustering x and y.
    Returns (n_rows, n_cols, rows_y, cols_x, debug)
    """
    xs = [c[0] for c in centers_xy]
    ys = [c[1] for c in centers_xy]

    gx = cluster_1d(xs, tol_x)
    gy = cluster_1d(ys, tol_y)

    cols_x = [float(np.median(g)) for g in gx]
    rows_y = [float(np.median(g)) for g in gy]

    # Sometimes over-splitting happens; we pick best candidate by closest count
    best = None
    for ncols in expected_cols_candidates:
        if len(cols_x) < ncols:
            continue
        # pick the densest ncols columns (by member count)
        col_counts = sorted([(len(gx[i]), cols_x[i]) for i in range(len(gx))], reverse=True)
        chosen_cols = sorted([x for _, x in col_counts[:ncols]])
        # recompute rows based on points close to these columns
        pts = [(x, y) for (x, y) in centers_xy if min(abs(x - cx) for cx in chosen_cols) <= tol_x]
        y_groups = cluster_1d([p[1] for p in pts], tol_y)
        cand_rows = [float(np.median(g)) for g in y_groups]
        nrows = len(cand_rows)

        if expected_rows is not None:
            score = abs(nrows - expected_rows) * 10 + abs(len(chosen_cols) - ncols)
        else:
            score = abs(len(chosen_cols) - ncols)

        if best is None or score < best[0]:
            best = (score, nrows, ncols, cand_rows, chosen_cols, pts)

    if best is None:
        return 0, 0, [], [], {"reason": "no grid candidate fits"}

    _, nrows, ncols, rows_y2, cols_x2, pts2 = best
    dbg = {
        "raw_cols": len(cols_x),
        "raw_rows": len(rows_y),
        "picked_cols": cols_x2,
        "picked_rows": rows_y2,
        "used_pts": len(pts2)
    }
    return nrows, ncols, rows_y2, cols_x2, dbg


def split_id_vs_questions(all_centers: List[Tuple[float, float, float]], img_h: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Split by Y: top cluster => ID, bottom cluster => questions.
    Works well for your sheet: ID at upper right, questions lower left.
    """
    pts = [(x, y) for (x, y, _) in all_centers]
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    if len(ys) < 10:
        return [], []
    # Otsu-like split by median
    mid = float(np.median(ys))
    top = [(x, y) for (x, y) in pts if y < mid]
    bot = [(x, y) for (x, y) in pts if y >= mid]

    # If split is wrong (ID is smaller cluster), prefer smaller cluster as ID
    if len(top) > len(bot):
        # choose smaller as ID
        id_pts, q_pts = (bot, top)
    else:
        id_pts, q_pts = (top, bot)

    return id_pts, q_pts


# -----------------------------
# Alignment (ORB + Homography)
# -----------------------------

def align_to_key(img_bgr: np.ndarray, key_gray: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """
    Align student image to key reference using ORB features.
    Returns warped BGR of size (w,h).
    """
    w, h = out_size
    g = gray(img_bgr)

    orb = cv2.ORB_create(2500)
    kp1, des1 = orb.detectAndCompute(g, None)
    kp2, des2 = orb.detectAndCompute(key_gray, None)

    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 20:
        return cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)

    warped = cv2.warpPerspective(img_bgr, H, (w, h))
    return warped


# -----------------------------
# Bubble scoring + X detection
# -----------------------------

def crop_cell(bin_img: np.ndarray, cx: float, cy: float, r: float, pad: float = 1.35) -> np.ndarray:
    h, w = bin_img.shape[:2]
    rr = max(6, int(r * pad))
    x1 = max(0, int(cx) - rr)
    y1 = max(0, int(cy) - rr)
    x2 = min(w, int(cx) + rr)
    y2 = min(h, int(cy) + rr)
    return bin_img[y1:y2, x1:x2]


def fill_ratio(cell_bin: np.ndarray) -> float:
    if cell_bin.size == 0:
        return 0.0
    # focus inner area to avoid circle border
    h, w = cell_bin.shape[:2]
    mh = int(h * 0.25)
    mw = int(w * 0.25)
    inner = cell_bin[mh:h - mh, mw:w - mw]
    if inner.size == 0:
        return 0.0
    return float(np.sum(inner > 0) / inner.size)


def x_mark_score(cell_bin: np.ndarray) -> float:
    """
    Detect X using edges + HoughLinesP.
    Returns score ~ how likely X exists (higher => more X).
    """
    if cell_bin.size == 0:
        return 0.0

    # edges on binary (already inv), use Canny on uint8
    img = (cell_bin > 0).astype(np.uint8) * 255
    edges = cv2.Canny(img, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                            minLineLength=max(10, int(min(cell_bin.shape) * 0.45)),
                            maxLineGap=5)
    if lines is None:
        return 0.0

    # count diagonal-ish long lines
    diag = 0
    for (x1, y1, x2, y2) in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 3:
            continue
        slope = dy / (dx + 1e-9)
        if 0.6 < abs(slope) < 1.8:
            length = math.hypot(dx, dy)
            if length >= max(12, int(min(cell_bin.shape) * 0.55)):
                diag += 1

    # Two diagonals usually => X
    return float(diag)


def choose_answer(cells_bin: List[np.ndarray],
                  labels: List[str],
                  blank_thr: float,
                  double_gap: float,
                  x_thr: float) -> Dict:
    """
    Smart choice:
      - Compute fill per option
      - Compute X score per option
      - Ignore options with X score >= x_thr
      - Pick max fill if >= blank_thr
      - If two top fills close => DOUBLE
    """
    fills = [fill_ratio(c) for c in cells_bin]
    xs = [x_mark_score(c) for c in cells_bin]

    valid = [i for i in range(len(labels)) if xs[i] < x_thr]
    if not valid:
        return {"answer": "?", "status": "BLANK", "fills": fills, "x": xs}

    # pick best among valid
    valid_sorted = sorted(valid, key=lambda i: fills[i], reverse=True)
    top = valid_sorted[0]
    top_fill = fills[top]
    second_fill = fills[valid_sorted[1]] if len(valid_sorted) > 1 else 0.0

    if top_fill < blank_thr:
        return {"answer": "?", "status": "BLANK", "fills": fills, "x": xs}

    if second_fill >= blank_thr and (top_fill - second_fill) < double_gap:
        return {"answer": "!", "status": "DOUBLE", "fills": fills, "x": xs}

    return {"answer": labels[top], "status": "OK", "fills": fills, "x": xs}


# -----------------------------
# Build template from Answer Key
# -----------------------------

def learn_template_from_key(key_bgr: np.ndarray,
                            debug: bool,
                            min_area: int,
                            max_area: int,
                            min_circ: float) -> Tuple[Template, Dict]:
    """
    Learn:
      - ID grid: must be 4 cols x 10 rows
      - Question grid: cols in {2,4,5}, rows = question count
    """
    h, w = key_bgr.shape[:2]
    centers = find_circle_centers(key_bgr, min_area=min_area, max_area=max_area, min_circ=min_circ)
    if len(centers) < 30:
        raise ValueError("عدد الفقاعات المكتشفة قليل جداً. ارفع DPI أو حسّن وضوح الصورة.")

    id_pts, q_pts = split_id_vs_questions(centers, h)

    # If split failed, fallback: pick ID by searching for 4x10 grid anywhere
    dbg = {"centers_total": len(centers), "split_id": len(id_pts), "split_q": len(q_pts)}

    # tolerance based on page size
    tol_x = w * 0.015
    tol_y = h * 0.015

    # --- ID grid (expected 10 rows, 4 cols)
    id_rows, id_cols, id_rows_y, id_cols_x, id_dbg = pick_grid_from_centers(
        id_pts, expected_rows=10, expected_cols_candidates=[4],
        tol_x=tol_x, tol_y=tol_y
    )

    # if still not 10x4, try alternative: search in all points
    if not (id_rows == 10 and id_cols == 4):
        id_rows, id_cols, id_rows_y, id_cols_x, id_dbg = pick_grid_from_centers(
            [(x, y) for (x, y, _) in centers], expected_rows=10, expected_cols_candidates=[4],
            tol_x=tol_x, tol_y=tol_y
        )

    if not (id_rows == 10 and id_cols == 4):
        raise ValueError("فشل تحديد شبكة كود الطالب (يجب أن تكون 4 أعمدة × 10 صفوف).")

    # collect id centers that fall near id col/row
    def nearest_idx(v, arr):
        return int(np.argmin([abs(v - a) for a in arr]))

    id_centers = []
    for (x, y, r) in centers:
        if min(abs(x - cx) for cx in id_cols_x) <= tol_x and min(abs(y - ry) for ry in id_rows_y) <= tol_y:
            id_centers.append((x, y))

    # --- Question grid (cols candidates 2/4/5)
    # Use q_pts if reasonable, else all (minus id region)
    q_base = q_pts if len(q_pts) >= 20 else [(x, y) for (x, y, _) in centers]

    q_rows, q_cols, q_rows_y, q_cols_x, q_dbg = pick_grid_from_centers(
        q_base, expected_rows=None, expected_cols_candidates=[2, 4, 5],
        tol_x=tol_x, tol_y=tol_y
    )
    # sanity: questions shouldn't be 4x10 (already ID). Need rows >= 5.
    if q_cols not in (2, 4, 5) or q_rows < 5:
        # try all centers but excluding those close to ID columns+rows area
        q2 = []
        for (x, y, _) in centers:
            # exclude ID points by proximity
            if min(abs(x - cx) for cx in id_cols_x) <= tol_x and min(abs(y - ry) for ry in id_rows_y) <= tol_y:
                continue
            q2.append((x, y))
        q_rows, q_cols, q_rows_y, q_cols_x, q_dbg = pick_grid_from_centers(
            q2, expected_rows=None, expected_cols_candidates=[2, 4, 5],
            tol_x=tol_x, tol_y=tol_y
        )

    if q_cols not in (2, 4, 5) or q_rows < 5:
        raise ValueError("فشل تحديد شبكة الأسئلة. تأكد أن صفحة الـ Key واضحة وكاملة.")

    # Build grids
    key_g = gray(key_bgr)

    id_grid = Grid(
        centers=id_centers,
        cols_x=sorted(id_cols_x),
        rows_y=sorted(id_rows_y),
        n_cols=4, n_rows=10
    )

    # question centers near q col/row
    q_centers = []
    for (x, y, r) in centers:
        if min(abs(x - cx) for cx in q_cols_x) <= tol_x and min(abs(y - ry) for ry in q_rows_y) <= tol_y:
            q_centers.append((x, y))

    q_grid = Grid(
        centers=q_centers,
        cols_x=sorted(q_cols_x),
        rows_y=sorted(q_rows_y),
        n_cols=int(q_cols),
        n_rows=int(q_rows)
    )

    tmpl = Template(
        key_size=(w, h),
        id_grid=id_grid,
        q_grid=q_grid,
        q_choices=int(q_cols),
        q_count=int(q_rows),
        key_ref_gray=key_g
    )

    dbg.update({
        "id_rows": id_rows, "id_cols": id_cols,
        "q_rows": q_rows, "q_cols": q_cols,
        "id_dbg": id_dbg,
        "q_dbg": q_dbg,
    })

    return tmpl, dbg


def overlay_debug(key_bgr: np.ndarray, tmpl: Template) -> np.ndarray:
    img = key_bgr.copy()
    # ID: red
    for cx in tmpl.id_grid.cols_x:
        cv2.line(img, (int(cx), 0), (int(cx), img.shape[0] - 1), (0, 0, 255), 2)
    for ry in tmpl.id_grid.rows_y:
        cv2.circle(img, (int(tmpl.id_grid.cols_x[0]), int(ry)), 4, (0, 0, 255), -1)

    # Q: blue vertical lines + green points
    for cx in tmpl.q_grid.cols_x:
        cv2.line(img, (int(cx), 0), (int(cx), img.shape[0] - 1), (255, 0, 0), 2)
    for (x, y) in tmpl.q_grid.centers:
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), 2)

    return img


# -----------------------------
# Extract key answers / student ID / student answers
# -----------------------------

def read_id_from_sheet(sheet_bgr_aligned: np.ndarray, tmpl: Template,
                       blank_thr: float, double_gap: float, x_thr: float,
                       bubble_r: float = 28.0) -> str:
    b = adaptive_bin_inv(gray(sheet_bgr_aligned))
    digits = []
    labels = [str(i) for i in range(10)]  # rows represent digits 0..9

    # Each column is a digit. For each digit column, choose one row (0..9).
    for col_i, cx in enumerate(tmpl.id_grid.cols_x):
        cells = []
        for row_i, ry in enumerate(tmpl.id_grid.rows_y):
            cell = crop_cell(b, cx, ry, bubble_r)
            cells.append(cell)

        res = choose_answer(cells, labels, blank_thr, double_gap, x_thr)
        # If OK => digit, else X
        digits.append(res["answer"] if res["status"] == "OK" else "X")

    return "".join(digits)


def extract_answers_from_sheet(sheet_bgr_aligned: np.ndarray, tmpl: Template,
                               blank_thr: float, double_gap: float, x_thr: float,
                               bubble_r: float = 28.0) -> Dict[int, Dict]:
    b = adaptive_bin_inv(gray(sheet_bgr_aligned))

    choices = ["A", "B", "C", "D", "E"][:tmpl.q_choices]
    out = {}

    for q_idx, ry in enumerate(tmpl.q_grid.rows_y, start=1):
        cells = []
        for cx in tmpl.q_grid.cols_x:
            cell = crop_cell(b, cx, ry, bubble_r)
            cells.append(cell)

        res = choose_answer(cells, choices, blank_thr, double_gap, x_thr)
        out[q_idx] = res

    return out


def build_answer_key_from_key_image(key_bgr: np.ndarray, tmpl: Template,
                                    blank_thr: float, double_gap: float, x_thr: float) -> Dict[int, str]:
    ans = extract_answers_from_sheet(key_bgr, tmpl, blank_thr, double_gap, x_thr)
    # only OK are considered true answers
    key = {}
    for q, res in ans.items():
        if res["status"] == "OK":
            key[q] = res["answer"]
    return key


# -----------------------------
# Roster loader
# -----------------------------

def load_roster(uploaded) -> Dict[str, str]:
    if uploaded is None:
        return {}
    name = (uploaded.name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # flexible column mapping
    possible_code = ["student_code", "code", "id", "studentid", "student_id"]
    possible_name = ["student_name", "name", "student", "fullname", "full_name"]

    code_col = next((c for c in possible_code if c in df.columns), None)
    name_col = next((c for c in possible_name if c in df.columns), None)

    if code_col is None or name_col is None:
        raise ValueError("ملف الطلاب يجب أن يحتوي عمود كود وعمود اسم (مثل: student_code, student_name).")

    codes = df[code_col].astype(str).str.strip()
    names = df[name_col].astype(str).str.strip()
    return dict(zip(codes, names))


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Hybrid OMR + Smart X", layout="wide")

    st.title("✅ Hybrid OMR + AI Rules (Smart X) — تدريب تلقائي من الـ Answer Key")

    # Settings
    st.session_state.setdefault("dpi", 300)

    with st.expander("⚙️ إعدادات عامة", expanded=True):
        dpi = st.selectbox("DPI للـ PDF", [200, 250, 300, 350, 400], index=2)
        st.session_state["dpi"] = dpi

        c1, c2, c3 = st.columns(3)
        with c1:
            min_area = st.number_input("min_area", 50, 5000, 120, 10)
            max_area = st.number_input("max_area", 500, 30000, 9000, 100)
        with c2:
            min_circ = st.slider("min_circularity", 0.10, 0.95, 0.34, 0.01)
        with c3:
            blank_thr = st.slider("Blank fill threshold", 0.01, 0.40, 0.14, 0.01)
            double_gap = st.slider("Double gap threshold (فرق الفقاعتين)", 0.00, 0.20, 0.03, 0.01)
            x_thr = st.slider("X threshold (عدد خطوط X)", 0.0, 5.0, 1.0, 0.5)

        debug = st.checkbox("Debug (عرض التفاصيل)", value=True)

    st.markdown("---")

    cA, cB, cC = st.columns(3)
    with cA:
        roster_file = st.file_uploader("📋 ملف الطلاب (Excel/CSV)", type=["xlsx", "xls", "csv"])
    with cB:
        key_file = st.file_uploader("🔑 Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    with cC:
        sheets_file = st.file_uploader("📄 أوراق الطلاب (PDF/صورة) — يمكن PDF متعدد الصفحات", type=["pdf", "png", "jpg", "jpeg"])

    st.markdown("---")

    # Step 1: Train
    st.subheader("1) 📌 تدريب تلقائي من الـ Answer Key")
    train_btn = st.button("🧠 تدريب/إعادة تدريب من Answer Key", type="primary")

    if "template" not in st.session_state:
        st.session_state["template"] = None
    if "answer_key" not in st.session_state:
        st.session_state["answer_key"] = None

    if train_btn:
        if key_file is None:
            st.error("ارفع ملف الـ Answer Key أولاً.")
        else:
            try:
                key_pil = pil_from_upload(key_file)
                key_bgr = to_bgr(key_pil)

                tmpl, dbg = learn_template_from_key(
                    key_bgr, debug=debug,
                    min_area=int(min_area), max_area=int(max_area), min_circ=float(min_circ)
                )
                st.session_state["template"] = tmpl

                # Extract answer key (BEFORE grading show it)
                answer_key = build_answer_key_from_key_image(key_bgr, tmpl, blank_thr, double_gap, x_thr)
                st.session_state["answer_key"] = answer_key

                st.success("✅ التدريب نجح.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("عدد الأسئلة المكتشف", str(tmpl.q_count))
                c2.metric("عدد الخيارات المكتشف", str(tmpl.q_choices))
                c3.metric("خانات كود الطالب", "4")
                c4.metric("صفوف كود الطالب", "10")

                if debug:
                    st.caption(f"Debug: {dbg}")

                # Show overlay
                over = overlay_debug(key_bgr, tmpl)
                st.image(cv2.cvtColor(over, cv2.COLOR_BGR2RGB), caption="Overlay (أحمر=كود، أزرق=أعمدة خيارات، أخضر=مراكز أسئلة)")

                # Show extracted Answer Key clearly
                st.subheader("🔎 Answer Key المستخرج (راجعه قبل التصحيح)")
                st.json({str(k): v for k, v in answer_key.items()})

                # download extracted answer key
                dfk = pd.DataFrame([{"question": k, "answer": v} for k, v in answer_key.items()])
                buf = io.BytesIO()
                dfk.to_csv(buf, index=False, encoding="utf-8-sig")
                st.download_button("⬇️ تنزيل Answer Key (CSV)", buf.getvalue(), file_name="answer_key_extracted.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ فشل التدريب التلقائي: {e}")
                if debug:
                    import traceback
                    st.code(traceback.format_exc())

    st.markdown("---")

    # Step 2: Grade
    st.subheader("2) ✅ التصحيح")
    grade_btn = st.button("🚀 ابدأ التصحيح", disabled=(st.session_state["template"] is None or st.session_state["answer_key"] is None))

    if grade_btn:
        if roster_file is None:
            st.error("ارفع ملف الطلاب (Roster) أولاً.")
            return
        if sheets_file is None:
            st.error("ارفع أوراق الطلاب أولاً.")
            return

        try:
            roster = load_roster(roster_file)
            tmpl: Template = st.session_state["template"]
            answer_key: Dict[int, str] = st.session_state["answer_key"]

            # Load student pages
            sheets_name = (sheets_file.name or "").lower()
            if sheets_name.endswith(".pdf"):
                pages = pdf_all_pages_bytes(sheets_file.getvalue(), dpi=int(st.session_state["dpi"]))
            else:
                pages = [Image.open(io.BytesIO(sheets_file.getvalue())).convert("RGB")]

            results = []
            with st.spinner("⏳ جاري التصحيح..."):
                for idx, p in enumerate(pages, start=1):
                    student_bgr = to_bgr(p)

                    # align
                    aligned = align_to_key(student_bgr, tmpl.key_ref_gray, tmpl.key_size)

                    # read id
                    sid = read_id_from_sheet(aligned, tmpl, blank_thr, double_gap, x_thr)
                    sname = roster.get(str(sid).strip(), "غير موجود")

                    # read answers
                    student_ans = extract_answers_from_sheet(aligned, tmpl, blank_thr, double_gap, x_thr)

                    correct = 0
                    total = len(answer_key)

                    for q, corr in answer_key.items():
                        res = student_ans.get(q, {"answer": "?", "status": "BLANK"})
                        if res["status"] == "OK" and res["answer"] == corr:
                            correct += 1

                    perc = (correct / total * 100.0) if total else 0.0

                    results.append({
                        "sheet_index": idx,
                        "student_code": sid,
                        "student_name": sname,
                        "score": correct,
                        "total_questions": total,
                        "percentage": round(perc, 2)
                    })

            dfres = pd.DataFrame(results)
            st.success("✅ اكتمل التصحيح.")
            st.dataframe(dfres, width="stretch", height=300)

            # export excel
            out = io.BytesIO()
            dfres.to_excel(out, index=False, engine="openpyxl")
            st.download_button(
                "⬇️ تحميل النتائج (Excel)",
                out.getvalue(),
                file_name="results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء التصحيح: {e}")
            if debug:
                import traceback
                st.code(traceback.format_exc())

    # Helpful note
    st.markdown("---")
    st.info(
        "ملاحظة مهمة:\n"
        "إذا كانت ورقة الطالب مائلة جدًا أو مصورة بزاوية قوية، المحاذاة قد تضعف.\n"
        "أفضل نتيجة: تصوير/سكنر واضح + الصفحة كاملة + DPI 300 أو أكثر."
    )


if __name__ == "__main__":
    main()
