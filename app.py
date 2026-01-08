# app.py
# ============================================================
# ✅ Smart OMR (Auto-learn from Answer Key) + Ignore X Marks
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
# File helpers
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
    centers: np.ndarray  # (rows, cols, 2) float
    rows: int
    cols: int


@dataclass
class LearnedTemplate:
    ref_bgr: np.ndarray
    ref_w: int
    ref_h: int
    id_grid: BubbleGrid  # 10 x digits
    q_grid: BubbleGrid   # num_q x choices
    num_q: int
    num_choices: int
    id_rows: int
    id_digits: int


# ==============================
# Preprocess (for learning bubble centers)
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
# Build grids
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

    # Ensure order
    row_meds = np.median(grid[:, :, 1], axis=1)
    col_meds = np.median(grid[:, :, 0], axis=0)
    grid = grid[np.argsort(row_meds)][:, np.argsort(col_meds)]

    return BubbleGrid(centers=grid, rows=rows, cols=cols)


def split_id_questions(centers: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = centers[:, 0]
    ys = centers[:, 1]

    # ID usually top-right
    id_mask = (xs > 0.55 * w) & (ys < 0.55 * h)
    # Questions usually lower-left
    q_mask = (xs < 0.55 * w) & (ys > 0.40 * h)

    id_centers = centers[id_mask]
    q_centers = centers[q_mask]

    # Fallback if strict
    if id_centers.shape[0] < 25:
        id_centers = centers[(xs > np.median(xs)) & (ys < np.median(ys))]
    if q_centers.shape[0] < 25:
        q_centers = centers[(xs < np.median(xs)) & (ys > np.median(ys))]

    return id_centers, q_centers


# ==============================
# ORB alignment (student -> key)
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

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return cv2.resize(student_bgr, (w, h)), False, len(good)

    warped = cv2.warpPerspective(student_bgr, H, (w, h))
    return warped, True, len(good)


# ==============================
# SMART Bubble reading:
# - Darkness (mean) chooses filled
# - X-mark detector excludes crossed bubbles
# - Policy: if X exists + a filled exists -> pick filled only
# ==============================
def bubble_roi(gray: np.ndarray, cx: int, cy: int, win: int = 18) -> np.ndarray:
    h, w = gray.shape[:2]
    x1 = max(0, cx - win); x2 = min(w, cx + win)
    y1 = max(0, cy - win); y2 = min(h, cy + win)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    rh, rw = roi.shape
    mh = int(rh * 0.20); mw = int(rw * 0.20)
    inner = roi[mh:rh-mh, mw:rw-mw]
    return inner if inner.size > 0 else roi


def x_mark_score(roi_gray: np.ndarray) -> float:
    g = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    edges = cv2.Canny(g, 50, 150)

    min_len = max(10, int(min(roi_gray.shape) * 0.45))
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=20, minLineLength=min_len, maxLineGap=3
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
    mean = float(np.mean(roi))
    std = float(np.std(roi))
    score = float(x_mark_score(roi))
    return {"mean": mean, "std": std, "xscore": score}


def pick_choice_smart(stats_list: List[dict],
                      labels: List[str],
                      blank_mean_thresh: float,
                      diff_thresh: float,
                      x_std_thresh: float,
                      x_score_thresh: float,
                      filled_mean_thresh: float) -> Tuple[str, str, List[bool]]:
    """
    Policy:
    - cancelled (X) excluded.
    - if any X exists and best (not-cancelled) looks filled -> choose it OK.
    - double evaluated only among not-cancelled.
    """
    cancelled = []
    means = []
    for s in stats_list:
        means.append(s["mean"])
        cancelled.append((s["std"] >= x_std_thresh) and (s["xscore"] >= x_score_thresh))

    cand = [i for i in range(len(labels)) if not cancelled[i]]
    if len(cand) == 0:
        return "?", "CANCELLED_ALL", cancelled

    cand_sorted = sorted(cand, key=lambda i: means[i])  # darker first
    best = cand_sorted[0]
    best_m = means[best]
    second_m = means[cand_sorted[1]] if len(cand_sorted) > 1 else 999.0

    if best_m > blank_mean_thresh:
        return "?", "BLANK", cancelled

    # ✅ Special policy for your case
    if any(cancelled) and best_m <= filled_mean_thresh:
        return labels[best], "OK", cancelled

    if (second_m - best_m) < diff_thresh:
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
# Learn template from Answer Key
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


def extract_answer_key(template: LearnedTemplate,
                       choices: List[str],
                       blank_mean_thresh: float,
                       diff_thresh: float,
                       x_std_thresh: float,
                       x_score_thresh: float,
                       filled_mean_thresh: float) -> Tuple[Dict[int, str], pd.DataFrame]:
    gray = cv2.cvtColor(template.ref_bgr, cv2.COLOR_BGR2GRAY)
    key = {}
    rows_dbg = []

    for r in range(template.q_grid.rows):
        stats = []
        for c in range(template.q_grid.cols):
            cx, cy = template.q_grid.centers[r, c]
            stats.append(bubble_stats(gray, int(cx), int(cy), win=18))

        ans, status, cancelled = pick_choice_smart(
            stats, choices,
            blank_mean_thresh=blank_mean_thresh,
            diff_thresh=diff_thresh,
            x_std_thresh=x_std_thresh,
            x_score_thresh=x_score_thresh,
            filled_mean_thresh=filled_mean_thresh
        )
        if status == "OK":
            key[r + 1] = ans

        rows_dbg.append(
            [r + 1, ans, status, cancelled] +
            [round(s["mean"], 1) for s in stats] +
            [round(s["std"], 1) for s in stats] +
            [round(s["xscore"], 2) for s in stats]
        )

    df_dbg = pd.DataFrame(
        rows_dbg,
        columns=(["Q", "Picked", "Status", "CancelledFlags"] +
                 [f"mean_{c}" for c in choices] +
                 [f"std_{c}" for c in choices] +
                 [f"x_{c}" for c in choices])
    )
    return key, df_dbg


def read_student_id(template: LearnedTemplate,
                    gray: np.ndarray,
                    blank_mean_thresh: float,
                    diff_thresh: float) -> Tuple[str, pd.DataFrame]:
    digits = []
    dbg_rows = []
    for c in range(template.id_grid.cols):
        means = []
        for r in range(template.id_grid.rows):
            cx, cy = template.id_grid.centers[r, c]
            roi = bubble_roi(gray, int(cx), int(cy), win=16)
            means.append(float(np.mean(roi)))

        labels = [str(i) for i in range(template.id_grid.rows)]
        idx = np.argsort(means)  # darker first
        best = int(idx[0]); second = int(idx[1]) if len(idx) > 1 else best
        best_m = means[best]; second_m = means[second]

        if best_m > blank_mean_thresh:
            digit, status = "X", "BLANK"
        elif (second_m - best_m) < diff_thresh:
            digit, status = "X", "DOUBLE"
        else:
            digit, status = labels[best], "OK"

        digits.append(digit)
        dbg_rows.append([c + 1, digit, status] + [round(m, 1) for m in means])

    df_dbg = pd.DataFrame(dbg_rows, columns=["DigitCol", "Picked", "Status"] + [str(i) for i in range(template.id_rows)])
    return "".join(digits), df_dbg


def read_student_answers(template: LearnedTemplate,
                         gray: np.ndarray,
                         choices: List[str],
                         blank_mean_thresh: float,
                         diff_thresh: float,
                         x_std_thresh: float,
                         x_score_thresh: float,
                         filled_mean_thresh: float) -> pd.DataFrame:
    out = []
    for r in range(template.q_grid.rows):
        stats = []
        for c in range(template.q_grid.cols):
            cx, cy = template.q_grid.centers[r, c]
            stats.append(bubble_stats(gray, int(cx), int(cy), win=18))

        ans, status, cancelled = pick_choice_smart(
            stats, choices,
            blank_mean_thresh=blank_mean_thresh,
            diff_thresh=diff_thresh,
            x_std_thresh=x_std_thresh,
            x_score_thresh=x_score_thresh,
            filled_mean_thresh=filled_mean_thresh
        )

        out.append(
            [r + 1, ans, status, cancelled] +
            [round(s["mean"], 1) for s in stats] +
            [round(s["std"], 1) for s in stats] +
            [round(s["xscore"], 2) for s in stats]
        )

    return pd.DataFrame(
        out,
        columns=(["Q", "Picked", "Status", "CancelledFlags"] +
                 [f"mean_{c}" for c in choices] +
                 [f"std_{c}" for c in choices] +
                 [f"x_{c}" for c in choices])
    )


# ==============================
# Streamlit UI
# ==============================
def main():
    st.set_page_config(page_title="Smart OMR (X Policy)", layout="wide")
    st.title("✅ Smart OMR: يتعلّم من الأنسر + يتجاهل الإجابة المشطوبة (X)")

    col1, col2, col3 = st.columns(3)
    with col1:
        roster_file = st.file_uploader("📋 ملف الطلاب (Excel/CSV)", type=["xlsx", "xls", "csv"])
    with col2:
        key_file = st.file_uploader("🔑 Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    with col3:
        sheets_file = st.file_uploader("📚 أوراق الطلاب (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])

    st.markdown("---")
    st.subheader("إعدادات النموذج (حسب ورقتكم)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        dpi = st.slider("DPI", 150, 400, 250, 10)
    with c2:
        id_rows = st.number_input("صفوف الكود (0-9)", 10, 15, 10, 1)
    with c3:
        id_digits = st.number_input("خانات الكود", 1, 12, 4, 1)
    with c4:
        num_q = st.number_input("عدد الأسئلة", 1, 200, 10, 1)
    with c5:
        num_choices = st.selectbox("عدد الخيارات", [4, 5, 6], index=0)

    choices = list("ABCDEF"[:int(num_choices)])

    st.subheader("تعلم مراكز الفقاعات (Bubble Detection)")
    d1, d2, d3 = st.columns(3)
    with d1:
        min_area = st.number_input("min_area", 20, 2000, 120, 10)
    with d2:
        max_area = st.number_input("max_area", 1000, 30000, 9000, 500)
    with d3:
        min_circ = st.slider("min_circularity", 0.30, 0.95, 0.55, 0.01)

    st.subheader("قراءة التظليل + كشف X (مع سياسة الشطب)")
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        blank_mean_thresh = st.slider("Blank mean threshold", 120, 240, 170, 1)
    with r2:
        diff_thresh = st.slider("Double diff threshold", 3, 60, 12, 1)
    with r3:
        x_std_thresh = st.slider("X std threshold", 5.0, 80.0, 25.0, 1.0)
    with r4:
        x_score_thresh = st.slider("X line score threshold", 0.0, 10.0, 1.2, 0.1)
    with r5:
        filled_mean_thresh = st.slider("Filled mean threshold (X + Filled => choose Filled)", 80, 200, 150, 1)

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

    # Load key
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

    # Extract answer key
    answer_key, df_key_dbg = extract_answer_key(
        template,
        choices=choices,
        blank_mean_thresh=float(blank_mean_thresh),
        diff_thresh=float(diff_thresh),
        x_std_thresh=float(x_std_thresh),
        x_score_thresh=float(x_score_thresh),
        filled_mean_thresh=float(filled_mean_thresh)
    )
    st.write("🔑 Answer Key المستخرج:")
    st.write(answer_key)

    if debug:
        st.markdown("---")
        st.subheader("Debug: Key learning")
        st.image(bgr_to_rgb(key_bgr), caption="Answer Key (Original)", width="stretch")
        st.image(dbg["bin_key"], caption="Binary (for detection only)", width="stretch")

        vis = key_bgr.copy()
        for (x, y) in dbg["id_centers"]:
            cv2.circle(vis, (int(x), int(y)), 7, (0, 0, 255), 2)
        for (x, y) in dbg["q_centers"]:
            cv2.circle(vis, (int(x), int(y)), 7, (0, 255, 0), 2)
        st.image(bgr_to_rgb(vis), caption="Red=ID centers, Green=Q centers", width="stretch")

        st.dataframe(df_key_dbg, width="stretch", height=360)

    # Grade
    st.markdown("---")
    if st.button("🚀 ابدأ التصحيح", type="primary", width="stretch"):
        pages = load_pages(read_bytes(sheets_file), sheets_file.name, dpi=int(dpi))
        if not pages:
            st.error("فشل قراءة أوراق الطلاب")
            return

        results = []
        debug_samples = []

        prog = st.progress(0)
        for i, pil_page in enumerate(pages, start=1):
            page_bgr = pil_to_bgr(pil_page)
            aligned, ok, good_matches = orb_align(page_bgr, template.ref_bgr)

            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

            student_code, df_id_dbg = read_student_id(
                template, gray,
                blank_mean_thresh=float(blank_mean_thresh),
                diff_thresh=float(diff_thresh)
            )
            student_code = str(student_code).zfill(int(id_digits))
            student_name = roster.get(student_code, "غير موجود")

            df_ans = read_student_answers(
                template, gray,
                choices=choices,
                blank_mean_thresh=float(blank_mean_thresh),
                diff_thresh=float(diff_thresh),
                x_std_thresh=float(x_std_thresh),
                x_score_thresh=float(x_score_thresh),
                filled_mean_thresh=float(filled_mean_thresh)
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

            if debug and len(debug_samples) < 2:
                debug_samples.append((i, aligned, df_id_dbg, df_ans))

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

        if debug and debug_samples:
            st.markdown("---")
            st.header("Debug Samples (أول صفحتين)")
            for page_no, aligned, df_id_dbg, df_ans in debug_samples:
                st.subheader(f"صفحة {page_no}")
                st.image(bgr_to_rgb(aligned), caption="Aligned to Answer Key", width="stretch")
                st.subheader("ID Means (الأقل = مظلل)")
                st.dataframe(df_id_dbg, width="stretch")
                st.subheader("Answers Stats (mean/std/x + cancelled flags)")
                st.dataframe(df_ans, width="stretch", height=360)


if __name__ == "__main__":
    main()
