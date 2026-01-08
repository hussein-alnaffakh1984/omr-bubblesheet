# app.py
# ============================================================
# ✅ Smart OMR (Auto-learn from Answer Key)
# ✅ KEY: Apply X-cancel then pick filled (fix Q5)
# ✅ STUDENT: Apply X-cancel then pick filled
# ✅ Policy: "لو شطب خيار واحد + ظلّل خيار آخر → اختر المظلّل فقط"
# ============================================================

import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image


# ==============================
# Helpers
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


def load_pages(file_bytes: bytes, filename: str, dpi: int = 250) -> List[Image.Image]:
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
# Data models
# ==============================
@dataclass
class BubbleGrid:
    centers: np.ndarray  # (rows, cols, 2) float32
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
# Bubble detection
# ==============================
def preprocess_binary_for_detection(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )
    bin_img = cv2.medianBlur(bin_img, 3)
    return bin_img


def find_bubble_centers(bin_img: np.ndarray,
                        min_area: int,
                        max_area: int,
                        min_circularity: float) -> np.ndarray:
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 1e-6:
            continue
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < min_circularity:
            continue
        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    if not centers:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(centers, dtype=np.int32)


# ==============================
# Grid building
# ==============================
def cluster_1d_equal_bins(values: np.ndarray, k: int) -> np.ndarray:
    idx = np.argsort(values)
    labels = np.zeros(len(values), dtype=np.int32)
    n = len(values)
    for j in range(k):
        s = int(j * n / k)
        e = int((j + 1) * n / k)
        labels[idx[s:e]] = j
    return labels


def build_grid(centers: np.ndarray, rows: int, cols: int) -> Optional[BubbleGrid]:
    if centers.shape[0] < int(rows * cols * 0.7):
        return None

    xs = centers[:, 0].astype(np.float32)
    ys = centers[:, 1].astype(np.float32)

    rlab = cluster_1d_equal_bins(ys, rows)
    clab = cluster_1d_equal_bins(xs, cols)

    grid = np.zeros((rows, cols, 2), dtype=np.float32)
    cnt = np.zeros((rows, cols), dtype=np.int32)

    for (x, y), r, c in zip(centers, rlab, clab):
        grid[r, c, 0] += x
        grid[r, c, 1] += y
        cnt[r, c] += 1

    for r in range(rows):
        for c in range(cols):
            if cnt[r, c] > 0:
                grid[r, c] /= cnt[r, c]
            else:
                grid[r, c] = (np.median(xs), np.median(ys))

    # Sort rows/cols robustly
    row_meds = np.median(grid[:, :, 1], axis=1)
    col_meds = np.median(grid[:, :, 0], axis=0)
    grid = grid[np.argsort(row_meds)][:, np.argsort(col_meds)]

    return BubbleGrid(centers=grid, rows=rows, cols=cols)


def split_id_questions(centers: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = centers[:, 0]
    ys = centers[:, 1]

    id_mask = (xs > 0.55 * w) & (ys < 0.55 * h)
    q_mask = (xs < 0.55 * w) & (ys > 0.40 * h)

    id_centers = centers[id_mask]
    q_centers = centers[q_mask]

    if id_centers.shape[0] < 25:
        id_centers = centers[(xs > np.median(xs)) & (ys < np.median(ys))]
    if q_centers.shape[0] < 25:
        q_centers = centers[(xs < np.median(xs)) & (ys > np.median(ys))]

    return id_centers, q_centers


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
# Bubble ROI + X detector + fill score
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
    inner = roi[mh:rh-mh, mw:rw-mw]
    return inner if inner.size > 0 else roi


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


def fill_score(roi_gray: np.ndarray) -> float:
    """
    Higher = more filled.
    Uses Otsu threshold on ROI then percent of black pixels.
    """
    if roi_gray.size < 10:
        return 0.0
    g = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(np.mean(th > 0))  # 0..1


def bubble_stats(gray: np.ndarray, cx: int, cy: int, win: int = 18) -> dict:
    roi = bubble_roi(gray, cx, cy, win=win)
    mean = float(np.mean(roi))
    std = float(np.std(roi))
    xscore = float(x_mark_score(roi))
    fscore = float(fill_score(roi))
    return {"mean": mean, "std": std, "xscore": xscore, "fill": fscore}


# ==============================
# Picker (used for KEY and STUDENT)
# ==============================
def pick_choice_x_cancel(stats_list: List[dict],
                         labels: List[str],
                         blank_fill_thresh: float,
                         double_fill_gap: float,
                         x_std_thresh: float,
                         x_score_thresh: float) -> Tuple[str, str, List[bool]]:
    """
    - cancelled if looks like X (std high + xscore high)
    - choose highest fill among not-cancelled
    - blank if best_fill < blank_fill_thresh
    - double if (best - second) < double_fill_gap (among not-cancelled)
    """
    cancelled = []
    fills = []
    for s in stats_list:
        fills.append(s["fill"])
        cancelled.append((s["std"] >= x_std_thresh) and (s["xscore"] >= x_score_thresh))

    cand = [i for i in range(len(labels)) if not cancelled[i]]
    if not cand:
        return "?", "CANCELLED_ALL", cancelled

    cand_sorted = sorted(cand, key=lambda i: fills[i], reverse=True)
    best = cand_sorted[0]
    best_f = fills[best]
    second_f = fills[cand_sorted[1]] if len(cand_sorted) > 1 else 0.0

    if best_f < blank_fill_thresh:
        return "?", "BLANK", cancelled

    if (best_f - second_f) < double_fill_gap:
        return "!", "DOUBLE", cancelled

    return labels[best], "OK", cancelled


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
# Learn template
# ==============================
def learn_template(key_bgr: np.ndarray,
                   id_rows: int, id_digits: int,
                   num_q: int, num_choices: int,
                   min_area: int, max_area: int, min_circ: float) -> Tuple[LearnedTemplate, Dict]:

    h, w = key_bgr.shape[:2]
    bin_key = preprocess_binary_for_detection(key_bgr)
    centers = find_bubble_centers(bin_key, min_area=min_area, max_area=max_area, min_circularity=min_circ)

    id_centers, q_centers = split_id_questions(centers, w, h)
    id_grid = build_grid(id_centers, rows=id_rows, cols=id_digits)
    q_grid = build_grid(q_centers, rows=num_q, cols=num_choices)

    if id_grid is None:
        raise ValueError("فشل تعلم شبكة الكود (ID). عدّل min_area/max_area/min_circularity أو ارفع DPI.")
    if q_grid is None:
        raise ValueError("فشل تعلم شبكة الأسئلة. عدّل min_area/max_area/min_circularity أو ارفع DPI.")

    template = LearnedTemplate(
        ref_bgr=key_bgr, ref_w=w, ref_h=h,
        id_grid=id_grid, q_grid=q_grid,
        num_q=num_q, num_choices=num_choices,
        id_rows=id_rows, id_digits=id_digits
    )
    dbg = {"bin_key": bin_key, "centers": centers, "id_centers": id_centers, "q_centers": q_centers}
    return template, dbg


# ==============================
# Extract KEY (WITH X-cancel ✅)
# ==============================
def extract_answer_key(template: LearnedTemplate,
                       choices: List[str],
                       blank_fill_thresh: float,
                       double_fill_gap: float,
                       x_std_thresh_key: float,
                       x_score_thresh_key: float) -> Tuple[Dict[int, str], pd.DataFrame]:
    gray = cv2.cvtColor(template.ref_bgr, cv2.COLOR_BGR2GRAY)
    key = {}
    rows_dbg = []

    for r in range(template.q_grid.rows):
        stats = []
        for c in range(template.q_grid.cols):
            cx, cy = template.q_grid.centers[r, c]
            stats.append(bubble_stats(gray, int(cx), int(cy), win=18))

        ans, status, cancelled = pick_choice_x_cancel(
            stats, choices,
            blank_fill_thresh=blank_fill_thresh,
            double_fill_gap=double_fill_gap,
            x_std_thresh=x_std_thresh_key,
            x_score_thresh=x_score_thresh_key
        )

        # حتى لو DOUBLE نخليه ظاهر بالديبغ، لكن لا ندخله بالمفتاح إلا OK
        if status == "OK":
            key[r + 1] = ans

        rows_dbg.append(
            [r + 1, ans, status, cancelled] +
            [round(s["fill"], 3) for s in stats] +
            [round(s["std"], 1) for s in stats] +
            [round(s["xscore"], 2) for s in stats]
        )

    df_dbg = pd.DataFrame(
        rows_dbg,
        columns=(["Q", "Picked", "Status", "CancelledFlags"] +
                 [f"fill_{c}" for c in choices] +
                 [f"std_{c}" for c in choices] +
                 [f"x_{c}" for c in choices])
    )
    return key, df_dbg


# ==============================
# Read student ID (fill only)
# ==============================
def read_student_id(template: LearnedTemplate,
                    gray: np.ndarray,
                    blank_fill_thresh: float,
                    double_fill_gap: float) -> Tuple[str, pd.DataFrame]:
    digits = []
    dbg_rows = []

    for c in range(template.id_grid.cols):
        fills = []
        for r in range(template.id_grid.rows):
            cx, cy = template.id_grid.centers[r, c]
            roi = bubble_roi(gray, int(cx), int(cy), win=16)
            fills.append(fill_score(roi))

        labels = [str(i) for i in range(template.id_grid.rows)]
        idx = np.argsort(fills)[::-1]  # highest fill first
        best = int(idx[0])
        second = int(idx[1]) if len(idx) > 1 else best
        best_f = fills[best]
        second_f = fills[second]

        if best_f < blank_fill_thresh:
            digit, status = "X", "BLANK"
        elif (best_f - second_f) < double_fill_gap:
            digit, status = "X", "DOUBLE"
        else:
            digit, status = labels[best], "OK"

        digits.append(digit)
        dbg_rows.append([c + 1, digit, status] + [round(f, 3) for f in fills])

    df_dbg = pd.DataFrame(dbg_rows, columns=["DigitCol", "Picked", "Status"] + [str(i) for i in range(template.id_rows)])
    return "".join(digits), df_dbg


# ==============================
# Read student answers (WITH X-cancel ✅)
# ==============================
def read_student_answers(template: LearnedTemplate,
                         gray: np.ndarray,
                         choices: List[str],
                         blank_fill_thresh: float,
                         double_fill_gap: float,
                         x_std_thresh: float,
                         x_score_thresh: float) -> pd.DataFrame:
    out = []
    for r in range(template.q_grid.rows):
        stats = []
        for c in range(template.q_grid.cols):
            cx, cy = template.q_grid.centers[r, c]
            stats.append(bubble_stats(gray, int(cx), int(cy), win=18))

        ans, status, cancelled = pick_choice_x_cancel(
            stats, choices,
            blank_fill_thresh=blank_fill_thresh,
            double_fill_gap=double_fill_gap,
            x_std_thresh=x_std_thresh,
            x_score_thresh=x_score_thresh
        )

        out.append(
            [r + 1, ans, status, cancelled] +
            [round(s["fill"], 3) for s in stats] +
            [round(s["std"], 1) for s in stats] +
            [round(s["xscore"], 2) for s in stats]
        )

    return pd.DataFrame(
        out,
        columns=(["Q", "Picked", "Status", "CancelledFlags"] +
                 [f"fill_{c}" for c in choices] +
                 [f"std_{c}" for c in choices] +
                 [f"x_{c}" for c in choices])
    )


# ==============================
# Streamlit UI
# ==============================
def main():
    st.set_page_config(page_title="Smart OMR (Fix Q5/Q6)", layout="wide")
    st.title("✅ Smart OMR (Fix Q5/Q6): الأنسر يفهم X + الطالب يفهم X")

    col1, col2, col3 = st.columns(3)
    with col1:
        roster_file = st.file_uploader("📋 ملف الطلاب (Excel/CSV)", type=["xlsx", "xls", "csv"])
    with col2:
        key_file = st.file_uploader("🔑 Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    with col3:
        sheets_file = st.file_uploader("📚 أوراق الطلاب (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])

    st.markdown("---")
    st.subheader("إعدادات الورقة")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        dpi = st.slider("DPI", 150, 450, 250, 10)
    with c2:
        id_rows = st.number_input("صفوف الكود (0-9)", 10, 15, 10, 1)
    with c3:
        id_digits = st.number_input("خانات الكود", 1, 12, 4, 1)
    with c4:
        num_q = st.number_input("عدد الأسئلة (مهم جداً)", 1, 300, 10, 1)
    with c5:
        num_choices = st.selectbox("عدد الخيارات", [4, 5, 6], index=0)

    choices = list("ABCDEF"[:int(num_choices)])

    st.subheader("تعلم مراكز الفقاعات (Bubble Detection)")
    d1, d2, d3 = st.columns(3)
    with d1:
        min_area = st.number_input("min_area", 20, 2500, 120, 10)
    with d2:
        max_area = st.number_input("max_area", 500, 40000, 9000, 500)
    with d3:
        min_circ = st.slider("min_circularity", 0.25, 0.95, 0.55, 0.01)

    st.subheader("Thresholds (Fill + X)")
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    with r1:
        blank_fill_thresh = st.slider("Blank fill threshold", 0.01, 0.60, 0.12, 0.01)
    with r2:
        double_fill_gap = st.slider("Double gap threshold", 0.01, 0.60, 0.10, 0.01)
    with r3:
        x_std_student = st.slider("X std (student)", 5.0, 80.0, 22.0, 1.0)
    with r4:
        x_score_student = st.slider("X score (student)", 0.0, 10.0, 1.2, 0.1)
    with r5:
        x_std_key = st.slider("X std (key)", 5.0, 120.0, 18.0, 1.0)
    with r6:
        x_score_key = st.slider("X score (key)", 0.0, 10.0, 0.9, 0.1)

    debug = st.checkbox("Debug", value=True)

    if not (roster_file and key_file and sheets_file):
        st.info("ارفع الملفات الثلاثة ثم ابدأ.")
        return

    # Load roster
    try:
        roster = load_roster(roster_file, id_digits=int(id_digits))
        st.success(f"✅ تم تحميل قائمة الطلاب: {len(roster)}")
    except Exception as e:
        st.error(f"خطأ ملف الطلاب: {e}")
        return

    # Load key page
    try:
        key_pages = load_pages(read_bytes(key_file), key_file.name, dpi=int(dpi))
        if not key_pages:
            st.error("فشل قراءة الأنسر")
            return
        key_bgr = pil_to_bgr(key_pages[0])
    except Exception as e:
        st.error(f"فشل فتح الأنسر: {e}")
        return

    # Learn template
    try:
        template, dbg = learn_template(
            key_bgr,
            id_rows=int(id_rows), id_digits=int(id_digits),
            num_q=int(num_q), num_choices=int(num_choices),
            min_area=int(min_area), max_area=int(max_area), min_circ=float(min_circ)
        )
        st.success("✅ تم تعلم القالب من الأنسر.")
    except Exception as e:
        st.error(f"❌ فشل تعلم القالب: {e}")
        return

    # Extract answer key (WITH X-cancel)
    answer_key, df_key_dbg = extract_answer_key(
        template,
        choices=choices,
        blank_fill_thresh=float(blank_fill_thresh),
        double_fill_gap=float(double_fill_gap),
        x_std_thresh_key=float(x_std_key),
        x_score_thresh_key=float(x_score_key),
    )

    st.markdown("### 🔑 Answer Key المستخرج")
    st.write(answer_key)

    if debug:
        st.markdown("---")
        st.subheader("Debug: Key extraction table (شوف Q5 و Q6 هنا)")
        st.dataframe(df_key_dbg, width="stretch", height=420)

    st.markdown("---")
    if st.button("🚀 ابدأ التصحيح", type="primary", width="stretch"):
        pages = load_pages(read_bytes(sheets_file), sheets_file.name, dpi=int(dpi))
        if not pages:
            st.error("فشل قراءة أوراق الطلاب")
            return

        results = []
        prog = st.progress(0)

        for i, pil_page in enumerate(pages, start=1):
            page_bgr = pil_to_bgr(pil_page)
            aligned, ok, good_matches = orb_align(page_bgr, template.ref_bgr)
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

            student_code, _ = read_student_id(
                template, gray,
                blank_fill_thresh=float(blank_fill_thresh),
                double_fill_gap=float(double_fill_gap),
            )
            student_code = str(student_code).zfill(int(id_digits))
            student_name = roster.get(student_code, "غير موجود")

            df_ans = read_student_answers(
                template, gray,
                choices=choices,
                blank_fill_thresh=float(blank_fill_thresh),
                double_fill_gap=float(double_fill_gap),
                x_std_thresh=float(x_std_student),
                x_score_thresh=float(x_score_student),
            )

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
            status = "ناجح ✓" if pct >= 50 else "راسب ✗"

            results.append({
                "page": i,
                "aligned_ok": ok,
                "good_matches": good_matches,
                "student_code": student_code,
                "student_name": student_name,
                "score": correct,
                "total": total,
                "percentage": round(pct, 2),
                "status": status,
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


if __name__ == "__main__":
    main()
