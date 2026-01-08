# -*- coding: utf-8 -*-
"""
Hybrid OMR (Circles) + Smart Rules (X-cancel) - Auto Template
✅ Detects bubbles even if empty (HoughCircles)
✅ Auto-detects ID grid (4x10) vs Questions grid (2/4/5 choices, variable rows)
✅ Handles: "X on one choice + fill another" => choose filled only
✅ Shows extracted Answer Key BEFORE grading
Streamlit 1.52+ (uses width="stretch")
"""

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
# Data models
# -----------------------------
@dataclass
class Grid:
    cols_x: List[float]          # sorted x centers
    rows_y: List[float]          # sorted y centers
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    bubble_r: float              # crop radius
    kind: str                    # "ID" or "Q"
    choices: Optional[List[str]] = None  # for Q grid

@dataclass
class Template:
    width: int
    height: int
    id_grid: Grid
    q_grid: Grid
    num_choices: int
    num_questions: int


# -----------------------------
# Utils: file -> image
# -----------------------------
def load_first_page(file_bytes: bytes, filename: str, dpi: int = 300) -> Image.Image:
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        if not pages:
            raise ValueError("PDF فارغ أو لم يتم تحويله.")
        return pages[0].convert("RGB")
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def bgr_from_pil(img: Image.Image) -> np.ndarray:
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# -----------------------------
# Core: circle detection
# -----------------------------
def detect_circles_hough(img_bgr: np.ndarray,
                         dp: float = 1.2,
                         min_dist: int = 22,
                         param1: int = 120,
                         param2: int = 26,
                         min_r: int = 10,
                         max_r: int = 40) -> List[Tuple[float, float, float]]:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)

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

    # إزالة تكرارات المراكز القريبة
    merged: List[Tuple[float, float, float]] = []
    for x, y, r in out:
        ok = True
        for mx, my, mr in merged:
            if (x - mx) ** 2 + (y - my) ** 2 < 10 ** 2:
                ok = False
                break
        if ok:
            merged.append((x, y, r))
    return merged


def median_spacing(vals: List[float]) -> float:
    if len(vals) < 2:
        return 30.0
    s = np.sort(np.array(vals, dtype=np.float32))
    dif = np.diff(s)
    dif = dif[dif > 1e-6]
    if len(dif) == 0:
        return 30.0
    return float(np.median(dif))


def cluster_1d_sorted(vals: List[float], gap: float) -> List[List[float]]:
    if not vals:
        return []
    s = sorted(vals)
    groups = [[s[0]]]
    for v in s[1:]:
        if abs(v - groups[-1][-1]) <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def robust_centers_1d(vals: List[float], expected: Optional[int] = None) -> List[float]:
    """
    تجميع x أو y إلى مراكز أعمدة/صفوف بدون sklearn.
    """
    if len(vals) < 3:
        return sorted(vals)

    sp = median_spacing(vals)
    gap = max(12.0, sp * 0.6)
    groups = cluster_1d_sorted(vals, gap=gap)
    centers = [float(np.median(g)) for g in groups if len(g) >= 2]  # تجاهل الضوضاء

    # إذا كان expected معروف، حاول تقليم/دمج بسيط
    centers = sorted(centers)
    if expected is not None and len(centers) != expected:
        # fallback: cv2.kmeans على 1D
        xs = np.array(vals, dtype=np.float32).reshape(-1, 1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
        _, _, c = cv2.kmeans(xs, expected, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        centers = sorted([float(v[0]) for v in c])
    return centers


def circles_in_column(circles: List[Tuple[float, float, float]], col_x: float, tol: float) -> List[Tuple[float, float, float]]:
    return [c for c in circles if abs(c[0] - col_x) <= tol]


def fit_grid_bbox(cols_x: List[float], rows_y: List[float], pad: float = 2.2, r: float = 18.0) -> Tuple[int, int, int, int]:
    x1 = int(min(cols_x) - pad * r)
    x2 = int(max(cols_x) + pad * r)
    y1 = int(min(rows_y) - pad * r)
    y2 = int(max(rows_y) + pad * r)
    return x1, y1, x2, y2


# -----------------------------
# Find ID grid (4 cols x 10 rows)
# -----------------------------
def find_id_grid(circles: List[Tuple[float, float, float]], img_shape: Tuple[int, int]) -> Grid:
    h, w = img_shape[:2]
    xs = [c[0] for c in circles]
    ys = [c[1] for c in circles]
    rs = [c[2] for c in circles]
    base_r = float(np.median(rs)) if rs else 18.0

    # مراكز أعمدة محتملة كثيرة
    x_centers = robust_centers_1d(xs, expected=None)
    if len(x_centers) < 4:
        raise ValueError("دوائر قليلة لا تسمح باكتشاف كود الطالب.")

    # نجرب كل توليفة 4 أعمدة ونبحث عن التي تعطي 10 صفوف متناسقة
    best = None
    x_centers_sorted = sorted(x_centers)

    # tolerance للأعمدة
    tol_x = max(12.0, base_r * 0.9)

    for i in range(len(x_centers_sorted) - 3):
        cols = x_centers_sorted[i:i+4]

        # اجمع دوائر هذه الأعمدة
        picked = []
        for cx in cols:
            picked.extend(circles_in_column(circles, cx, tol=tol_x))

        if len(picked) < 4 * 8:
            continue

        # صفوف من y
        rows = robust_centers_1d([c[1] for c in picked], expected=10)
        if len(rows) != 10:
            continue

        # قياس جودة: كم دائرة تقع قرب تقاطعات الشبكة
        tol_y = max(12.0, base_r * 1.0)
        score = 0
        for cx in cols:
            for ry in rows:
                # هل يوجد دائرة قرب (cx, ry) ؟
                ok = False
                for (x, y, r) in picked:
                    if abs(x - cx) <= tol_x and abs(y - ry) <= tol_y:
                        ok = True
                        break
                score += 1 if ok else 0

        # شرط: يجب أن يغطي أغلب 40 تقاطع
        if score < 32:
            continue

        # تفضيل: ID غالباً كتلة طولية واضحة (ارتفاع كبير) وموقعها ليس داخل الأسئلة
        bbox = fit_grid_bbox(cols, rows, r=base_r)
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        # هدفنا أعلى score ثم أصغر area (لأن ID كتلة أصغر من الأسئلة)
        key = (score, -area)
        if best is None or key > best[0]:
            best = (key, cols, rows, bbox, base_r)

    if best is None:
        raise ValueError("لم أستطع تحديد شبكة كود الطالب 4×10. ارفع DPI أو حسّن وضوح الورقة.")

    _, cols, rows, bbox, r = best
    return Grid(cols_x=cols, rows_y=rows, bbox=bbox, bubble_r=r, kind="ID")


# -----------------------------
# Remove circles inside bbox
# -----------------------------
def filter_out_bbox(circles: List[Tuple[float, float, float]], bbox: Tuple[int, int, int, int]) -> List[Tuple[float, float, float]]:
    x1, y1, x2, y2 = bbox
    out = []
    for c in circles:
        if x1 <= c[0] <= x2 and y1 <= c[1] <= y2:
            continue
        out.append(c)
    return out


# -----------------------------
# Find Questions grid (k cols in {2,4,5}, rows variable)
# -----------------------------
def find_q_grid(circles: List[Tuple[float, float, float]], img_shape: Tuple[int, int], k_candidates=(2,4,5)) -> Tuple[Grid, int]:
    h, w = img_shape[:2]
    xs = [c[0] for c in circles]
    ys = [c[1] for c in circles]
    rs = [c[2] for c in circles]
    base_r = float(np.median(rs)) if rs else 18.0

    if len(circles) < 30:
        raise ValueError("دوائر الأسئلة قليلة جداً. تأكد أنك رفعت صفحة الأسئلة/الأنسر الصحيحة أو ارفع DPI.")

    best = None

    for k in k_candidates:
        # KMeans 1D على X (بدون sklearn) بواسطة cv2.kmeans
        data = np.array(xs, dtype=np.float32).reshape(-1, 1)
        if len(data) < k * 6:
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.01)
        compactness, labels, centers_x = cv2.kmeans(
            data, k, None, criteria, 6, cv2.KMEANS_PP_CENTERS
        )
        cols = sorted([float(v[0]) for v in centers_x])

        # استبعاد حلول غير منطقية (أعمدة قريبة جداً)
        dif = np.diff(cols)
        if len(dif) and float(np.min(dif)) < max(18.0, base_r * 1.2):
            continue

        # احسب صفوف من Y على كامل الدوائر (لكن مع فلترة بسيطة: قرب الأعمدة)
        tol_x = max(12.0, base_r * 1.0)
        picked = []
        for cx in cols:
            picked.extend(circles_in_column(circles, cx, tol=tol_x))

        if len(picked) < k * 6:
            continue

        # rows: تجميع Y تلقائياً
        rows_guess = robust_centers_1d([c[1] for c in picked], expected=None)
        # صفوف أسئلة لازم تكون >=5
        if len(rows_guess) < 5:
            continue

        # حدد عدد الأسئلة (rows) من توزيع Y: نأخذ "مراكز" بعد clustering بدون expected
        # ولتحسين الثبات: إذا rows كبيرة جداً بسبب ضوضاء، استعمل expected قريب من (len(picked)/k)
        est = int(round(len(picked)/k))
        if est >= 5 and est <= 200 and abs(len(rows_guess) - est) > 8:
            # إعادة تقدير بـ kmeans على Y
            ydata = np.array([c[1] for c in picked], dtype=np.float32).reshape(-1, 1)
            criteria2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.01)
            _, _, cy = cv2.kmeans(ydata, est, None, criteria2, 3, cv2.KMEANS_PP_CENTERS)
            rows_guess = sorted([float(v[0]) for v in cy])

        # bbox
        bbox = fit_grid_bbox(cols, rows_guess, r=base_r)

        # قياس جودة: كثافة تقاطعات موجودة
        tol_y = max(12.0, base_r * 1.0)
        hit = 0
        total = len(cols) * len(rows_guess)
        # لا نحسب كل التقاطعات لو كان العدد كبير جداً (لتسريع)
        step = 1 if total <= 300 else max(1, total // 300)
        idx = 0
        for cx in cols:
            for ry in rows_guess:
                idx += 1
                if idx % step != 0:
                    continue
                ok = False
                for (x, y, r) in picked:
                    if abs(x - cx) <= tol_x and abs(y - ry) <= tol_y:
                        ok = True
                        break
                hit += 1 if ok else 0

        # score: hit ratio - compactness penalty
        hit_ratio = hit / max(1, (total/step))
        score = hit_ratio - (float(compactness) / (1e7))

        if best is None or score > best[0]:
            best = (score, cols, rows_guess, bbox, base_r, k)

    if best is None:
        raise ValueError("فشل اكتشاف شبكة الأسئلة (2/4/5). ارفع DPI أو تأكد أن الصفحة كاملة بدون قص.")

    score, cols, rows, bbox, r, k = best

    # تقدير bubble_r من تباعد الأعمدة
    if len(cols) >= 2:
        col_spacing = float(np.median(np.diff(cols)))
        bubble_r = float(np.clip(col_spacing * 0.33, 10.0, 40.0))
    else:
        bubble_r = r

    choices = list("ABCDE")[:k]
    return Grid(cols_x=cols, rows_y=rows, bbox=bbox, bubble_r=bubble_r, kind="Q", choices=choices), k


# -----------------------------
# Preprocess for fill + X
# -----------------------------
def crop_roi(gray: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    h, w = gray.shape[:2]
    x1 = int(max(0, cx - r))
    x2 = int(min(w, cx + r))
    y1 = int(max(0, cy - r))
    y2 = int(min(h, cy + r))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1,1), dtype=np.uint8)
    return gray[y1:y2, x1:x2]


def fill_ratio(gray_roi: np.ndarray) -> float:
    """
    نسبة التظليل داخل الدائرة (تقريبياً)
    """
    if gray_roi.size < 25:
        return 0.0

    # binarize: foreground = dark ink
    g = cv2.GaussianBlur(gray_roi, (3,3), 0)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 21, 7)

    h, w = b.shape[:2]
    # inner region (avoid circle border)
    mh = int(h * 0.22)
    mw = int(w * 0.22)
    inner = b[mh:h-mh, mw:w-mw]
    if inner.size == 0:
        return 0.0
    return float(np.mean(inner > 0))


def x_cancel_score(gray_roi: np.ndarray) -> float:
    """
    يحاول اكتشاف وجود X داخل الفقاعة (خطين قطريين).
    نستخدم Canny + HoughLinesP ونحسب عدد خطوط مائلة واضحة.
    """
    if gray_roi.size < 25:
        return 0.0

    g = cv2.GaussianBlur(gray_roi, (3,3), 0)
    edges = cv2.Canny(g, 60, 160)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=18,
                            minLineLength=max(10, int(min(gray_roi.shape)*0.55)),
                            maxLineGap=6)
    if lines is None:
        return 0.0

    diag = 0
    for l in lines[:,0,:]:
        x1,y1,x2,y2 = l
        dx = x2 - x1
        dy = y2 - y1
        ang = abs(math.degrees(math.atan2(dy, dx)))
        # diagonal near 45 or 135
        if 25 <= ang <= 65 or 115 <= ang <= 155:
            diag += 1

    # score scaled
    return float(diag)


def pick_answer_for_row(fills: List[float], x_scores: List[float], choices: List[str],
                        blank_thr: float = 0.16,
                        double_gap: float = 1.35,
                        x_thr: float = 1.0) -> Dict:
    """
    Rules:
    - If a choice has X (x_score>=x_thr), treat it as cancelled unless it's the only marked one.
    - Pick highest fill among NOT-cancelled.
    - If highest < blank_thr => BLANK
    - If second is close => DOUBLE
    """
    n = len(fills)
    idx_sorted = sorted(range(n), key=lambda i: fills[i], reverse=True)

    # mark cancelled
    cancelled = [xs >= x_thr for xs in x_scores]

    # candidate indices: not cancelled
    candidates = [i for i in idx_sorted if not cancelled[i]]

    # if all cancelled, fallback to normal highest fill
    if not candidates:
        candidates = idx_sorted[:]

    top = candidates[0]
    top_fill = fills[top]
    second = candidates[1] if len(candidates) > 1 else None
    second_fill = fills[second] if second is not None else 0.0

    if top_fill < blank_thr:
        return {"answer": "?", "status": "BLANK", "fills": fills, "x": x_scores, "cancelled": cancelled}

    # double check
    if second is not None and second_fill >= blank_thr:
        ratio = top_fill / (second_fill + 1e-9)
        if ratio < double_gap:
            return {"answer": "!", "status": "DOUBLE", "fills": fills, "x": x_scores, "cancelled": cancelled}

    return {"answer": choices[top], "status": "OK", "fills": fills, "x": x_scores, "cancelled": cancelled}


# -----------------------------
# Read ID + Q answers using template
# -----------------------------
def read_id(img_bgr: np.ndarray, id_grid: Grid, debug: bool = False) -> str:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    digits = []
    # 10 rows -> digits 0..9
    digit_choices = [str(i) for i in range(10)]
    for cx in id_grid.cols_x:
        fills = []
        xs = []
        for ry in id_grid.rows_y:
            roi = crop_roi(gray, cx, ry, id_grid.bubble_r)
            fills.append(fill_ratio(roi))
            xs.append(x_cancel_score(roi))
        # للـ ID: لا نطبق إلغاء X بقوة، فقط أعلى fill
        idx = int(np.argmax(np.array(fills)))
        if fills[idx] < 0.14:
            digits.append("X")
        else:
            digits.append(digit_choices[idx])
    return "".join(digits)


def read_answers(img_bgr: np.ndarray, q_grid: Grid,
                 blank_thr: float, double_gap: float, x_thr: float) -> Dict[int, Dict]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    answers: Dict[int, Dict] = {}
    choices = q_grid.choices or list("ABCD")

    # rows are ordered top->bottom, question numbers start at 1
    qnum = 1
    for ry in q_grid.rows_y:
        fills = []
        xs = []
        for cx in q_grid.cols_x:
            roi = crop_roi(gray, cx, ry, q_grid.bubble_r)
            fills.append(fill_ratio(roi))
            xs.append(x_cancel_score(roi))
        res = pick_answer_for_row(
            fills, xs, choices=choices,
            blank_thr=blank_thr, double_gap=double_gap, x_thr=x_thr
        )
        answers[qnum] = res
        qnum += 1
    return answers


# -----------------------------
# Overlay drawing
# -----------------------------
def overlay_debug(img_bgr: np.ndarray, tmpl: Template) -> np.ndarray:
    out = img_bgr.copy()

    # draw ID grid (red)
    x1,y1,x2,y2 = tmpl.id_grid.bbox
    cv2.rectangle(out, (x1,y1), (x2,y2), (0,0,255), 3)
    for cx in tmpl.id_grid.cols_x:
        cv2.line(out, (int(cx), 0), (int(cx), out.shape[0]), (0,0,255), 2)

    # draw Q grid (green + blue columns)
    x1,y1,x2,y2 = tmpl.q_grid.bbox
    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 3)
    for cx in tmpl.q_grid.cols_x:
        cv2.line(out, (int(cx), 0), (int(cx), out.shape[0]), (255,0,0), 2)
    for ry in tmpl.q_grid.rows_y:
        cv2.circle(out, (int(tmpl.q_grid.cols_x[0]), int(ry)), 3, (0,255,0), -1)

    return out


# -----------------------------
# Roster loading
# -----------------------------
def load_roster(file) -> Dict[str, str]:
    if file is None:
        return {}
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "student_code" not in df.columns or "student_name" not in df.columns:
        raise ValueError("ملف الطلاب يجب أن يحتوي عمودين: student_code و student_name")
    codes = df["student_code"].astype(str).str.strip()
    names = df["student_name"].astype(str).str.strip()
    return dict(zip(codes, names))


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Hybrid OMR + AI Rules", layout="wide")
    st.title("✅ Hybrid OMR + AI Rules (Auto Template)")
    st.caption("يكتشف كود الطالب والأسئلة تلقائياً (بدون تعيين يدوي) + قاعدة إلغاء X")

    # session state
    if "template" not in st.session_state:
        st.session_state.template = None
    if "answer_key" not in st.session_state:
        st.session_state.answer_key = None

    # uploads
    c1, c2, c3 = st.columns(3)
    with c1:
        roster_file = st.file_uploader("📋 ملف الطلاب (Excel/CSV) - student_code, student_name", type=["xlsx","xls","csv"])
    with c2:
        key_file = st.file_uploader("🔑 Answer Key (PDF/صورة)", type=["pdf","png","jpg","jpeg"])
    with c3:
        sheets_file = st.file_uploader("🧾 أوراق الطلاب (PDF/صورة)", type=["pdf","png","jpg","jpeg"])

    dpi = st.selectbox("DPI للـ PDF", [200, 250, 300, 350, 400], index=2)
    debug = st.checkbox("Debug", value=True)

    st.divider()

    # thresholds
    st.subheader("Thresholds (Fill + X)")
    t1, t2, t3 = st.columns(3)
    with t1:
        blank_thr = st.slider("Blank fill threshold", 0.05, 0.35, 0.16, 0.01)
    with t2:
        double_gap = st.slider("Double gap ratio", 1.05, 2.00, 1.35, 0.05)
    with t3:
        x_thr = st.slider("X cancel score threshold", 0.0, 3.0, 1.0, 0.1)

    st.divider()

    # TRAIN button (from key)
    train_col1, train_col2 = st.columns([1,2])
    with train_col1:
        do_train = st.button("🎯 تدريب/إعادة تدريب من Answer Key", type="primary")
    with train_col2:
        st.info("التدريب يثبت القالب (Template). تغيير السلايدر لا يعيد التدريب إلا إذا ضغطت الزر.")

    if do_train:
        if key_file is None:
            st.error("ارفع Answer Key أولاً.")
        else:
            try:
                key_img = load_first_page(key_file.getvalue(), key_file.name, dpi=dpi)
                key_bgr = bgr_from_pil(key_img)

                circles = detect_circles_hough(
                    key_bgr,
                    dp=1.2, min_dist=22,
                    param1=120, param2=26,  # إذا ما يكشف كفاية: 24
                    min_r=10, max_r=40
                )

                if len(circles) < 40:
                    raise ValueError("عدد الدوائر المكتشفة قليل جداً. جرّب DPI أعلى أو قلّل param2 (مثلاً 24).")

                # 1) ID grid
                id_grid = find_id_grid(circles, key_bgr.shape)

                # 2) Q grid from remaining circles
                remaining = filter_out_bbox(circles, id_grid.bbox)
                q_grid, k = find_q_grid(remaining, key_bgr.shape, k_candidates=(2,4,5))

                # Build template
                tmpl = Template(
                    width=key_bgr.shape[1],
                    height=key_bgr.shape[0],
                    id_grid=id_grid,
                    q_grid=q_grid,
                    num_choices=k,
                    num_questions=len(q_grid.rows_y)
                )
                st.session_state.template = tmpl

                # Extract Answer Key (from key page itself)
                key_answers = read_answers(key_bgr, tmpl.q_grid, blank_thr, double_gap, x_thr)
                # Only keep OK answers (and also keep DOUBLE/BLANK for review)
                extracted = {}
                for q, r in key_answers.items():
                    extracted[q] = r["answer"]

                st.session_state.answer_key = extracted

                st.success(f"✅ التدريب نجح | خيارات={tmpl.num_choices} | أسئلة={tmpl.num_questions} | ID=4×10")

                if debug:
                    ov = overlay_debug(key_bgr, tmpl)
                    st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption="Overlay: ID=أحمر | أعمدة الخيارات=أزرق | صندوق الأسئلة=أخضر", width="stretch")

                # Show Answer Key before grading
                st.subheader("🔎 Answer Key المستخرج (تأكد قبل التصحيح)")
                dfk = pd.DataFrame([{"Q": q, "Key": a} for q, a in extracted.items()])
                st.dataframe(dfk, width="stretch", height=380)

            except Exception as e:
                st.error(f"❌ فشل التدريب: {e}")

    st.divider()

    # GRADING
    st.subheader("✅ التصحيح")
    if st.button("🚀 ابدأ التصحيح", type="primary"):
        if st.session_state.template is None or st.session_state.answer_key is None:
            st.error("لازم تدريب ناجح أولاً من Answer Key.")
            return
        if roster_file is None:
            st.error("ارفع ملف الطلاب أولاً.")
            return
        if sheets_file is None:
            st.error("ارفع ورقة الطالب/الطلاب أولاً.")
            return

        try:
            roster = load_roster(roster_file)
            tmpl: Template = st.session_state.template
            answer_key: Dict[int, str] = st.session_state.answer_key

            # Load student sheet image
            sheet_img = load_first_page(sheets_file.getvalue(), sheets_file.name, dpi=dpi)
            sheet_bgr = bgr_from_pil(sheet_img)

            # NOTE: نفترض نفس الأبعاد (مثل Remark) - لو اختلاف بسيط نعمل resize إلى قالب التدريب
            sheet_bgr = cv2.resize(sheet_bgr, (tmpl.width, tmpl.height), interpolation=cv2.INTER_AREA)

            # Read ID + answers
            student_id = read_id(sheet_bgr, tmpl.id_grid)
            student_name = roster.get(str(student_id).strip(), "غير موجود")

            student_answers = read_answers(sheet_bgr, tmpl.q_grid, blank_thr, double_gap, x_thr)

            # Score
            correct = 0
            total = min(tmpl.num_questions, len(answer_key))
            details = []
            for q in range(1, total+1):
                key_a = answer_key.get(q, "?")
                stu = student_answers.get(q, {"answer": "?"})
                stu_a = stu["answer"]
                ok = (stu_a == key_a)
                correct += 1 if ok else 0
                details.append({
                    "Q": q,
                    "Key": key_a,
                    "Student": stu_a,
                    "Status": stu.get("status",""),
                    "Correct": "✓" if ok else "✗"
                })

            pct = (correct / total * 100) if total else 0.0

            st.success(f"✅ تم التصحيح | ID={student_id} | الاسم={student_name} | النتيجة={correct}/{total} ({pct:.1f}%)")

            if debug:
                ov2 = overlay_debug(sheet_bgr, tmpl)
                st.image(cv2.cvtColor(ov2, cv2.COLOR_BGR2RGB), caption="Overlay على ورقة الطالب", width="stretch")

            st.subheader("📋 تفاصيل الإجابات")
            dfd = pd.DataFrame(details)
            st.dataframe(dfd, width="stretch", height=420)

            # Export Excel
            out_df = pd.DataFrame([{
                "student_code": student_id,
                "student_name": student_name,
                "score": correct,
                "total": total,
                "percentage": pct
            }])
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                out_df.to_excel(writer, index=False, sheet_name="summary")
                dfd.to_excel(writer, index=False, sheet_name="details")

            st.download_button(
                "⬇️ تحميل النتائج (Excel)",
                data=buf.getvalue(),
                file_name="results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch"
            )

        except Exception as e:
            st.error(f"❌ خطأ أثناء التصحيح: {e}")


if __name__ == "__main__":
    main()
