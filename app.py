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
# Helpers: file reading
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
# Template data models
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
    id_grid: BubbleGrid      # 10 x 4 (0..9 rows, 4 digits cols)
    q_grid: BubbleGrid       # 10 x 4 (Q1..Q10 rows, A..D cols)
    num_q: int
    num_choices: int
    id_rows: int
    id_digits: int


# ==============================
# Image processing
# ==============================
def preprocess_binary(bgr: np.ndarray) -> np.ndarray:
    """Binary image for bubble detection & fill. Marks become white (255)."""
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


# ==============================
# Bubble detection (contour circularity)
# ==============================
def find_bubbles_centers(bin_img: np.ndarray,
                        min_area: int = 80,
                        max_area: int = 8000,
                        min_circularity: float = 0.55) -> np.ndarray:
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

    centers = np.array(centers, dtype=np.int32)
    return centers


# ==============================
# Build grid from centers by clustering rows/cols
# ==============================
def cluster_1d_equal_bins(values: np.ndarray, k: int) -> np.ndarray:
    """Assign labels 0..k-1 by sorting and splitting into equal-count bins."""
    idx = np.argsort(values)
    labels = np.zeros(len(values), dtype=np.int32)
    n = len(values)
    for j in range(k):
        s = int(j * n / k)
        e = int((j + 1) * n / k)
        labels[idx[s:e]] = j
    return labels


def build_grid(centers: np.ndarray, rows: int, cols: int) -> Optional[BubbleGrid]:
    # Need enough points to be meaningful
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

    # average
    for r in range(rows):
        for c in range(cols):
            if cnt[r, c] > 0:
                grid[r, c] /= cnt[r, c]
            else:
                # fallback: use median of existing points
                grid[r, c] = (np.median(xs), np.median(ys))

    # Ensure rows are top->bottom and cols left->right:
    # (equal-bin already gives order, but keep safe by sorting row medians)
    row_meds = np.median(grid[:, :, 1], axis=1)
    col_meds = np.median(grid[:, :, 0], axis=0)
    row_order = np.argsort(row_meds)
    col_order = np.argsort(col_meds)
    grid = grid[row_order][:, col_order]

    return BubbleGrid(centers=grid, rows=rows, cols=cols)


# ==============================
# Auto split ID vs Questions (based on your sheet layout)
# ID is at TOP-RIGHT. Questions at LOWER-LEFT.
# ==============================
def split_id_and_questions(centers: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    if centers.shape[0] == 0:
        return centers, centers

    xs = centers[:, 0]
    ys = centers[:, 1]

    # TOP-RIGHT heuristic region for ID grid
    id_mask = (xs > 0.55 * w) & (ys < 0.55 * h)
    id_centers = centers[id_mask]

    # Questions likely in left + lower region
    q_mask = (xs < 0.55 * w) & (ys > 0.40 * h)
    q_centers = centers[q_mask]

    # Fallbacks if masks too strict
    if id_centers.shape[0] < 20:
        # take right half & top half
        id_mask = (xs > np.median(xs)) & (ys < np.median(ys))
        id_centers = centers[id_mask]
    if q_centers.shape[0] < 20:
        # take left half & lower half
        q_mask = (xs < np.median(xs)) & (ys > np.median(ys))
        q_centers = centers[q_mask]

    return id_centers, q_centers


# ==============================
# ORB alignment: student page -> key reference
# ==============================
def orb_align(student_bgr: np.ndarray, ref_bgr: np.ndarray) -> Tuple[np.ndarray, bool, int]:
    """Align student to reference using ORB + homography (RANSAC)."""
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
# Fill measurement (window around center)
# ==============================
def fill_ratio(bin_img: np.ndarray, cx: int, cy: int, win: int = 18) -> float:
    hh, ww = bin_img.shape[:2]
    x1 = max(0, cx - win)
    x2 = min(ww, cx + win)
    y1 = max(0, cy - win)
    y2 = min(hh, cy + win)
    patch = bin_img[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    # inner (avoid ring a bit)
    ph, pw = patch.shape[:2]
    mh = int(ph * 0.25)
    mw = int(pw * 0.25)
    inner = patch[mh:ph - mh, mw:pw - mw]
    if inner.size == 0:
        inner = patch
    return float(np.sum(inner > 0)) / float(inner.size)


def pick_choice(fills: List[float], labels: List[str], min_fill: float, ratio: float) -> Tuple[str, str]:
    idx = np.argsort(fills)[::-1]
    top = int(idx[0])
    top_fill = fills[top]
    second = fills[int(idx[1])] if len(idx) > 1 else 0.0

    if top_fill < min_fill:
        return "?", "BLANK"
    if second >= min_fill and (top_fill / (second + 1e-9)) < ratio:
        return "!", "DOUBLE"
    return labels[top], "OK"


# ==============================
# Roster
# ==============================
def load_roster(file) -> Dict[str, str]:
    if file.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "student_code" not in df.columns or "student_name" not in df.columns:
        raise ValueError("ملف الطلاب يجب أن يحتوي عمودين: student_code و student_name")

    codes = df["student_code"].astype(str).str.strip().str.replace(".0", "", regex=False)
    names = df["student_name"].astype(str).str.strip()
    return dict(zip(codes, names))


# ==============================
# Learn template from Answer Key
# ==============================
def learn_template_from_answer_key(key_bgr: np.ndarray,
                                   id_rows: int,
                                   id_digits: int,
                                   num_q: int,
                                   num_choices: int,
                                   min_area: int,
                                   max_area: int,
                                   min_circ: float) -> Tuple[LearnedTemplate, Dict]:
    dbg = {}
    h, w = key_bgr.shape[:2]

    key_bin = preprocess_binary(key_bgr)
    centers = find_bubbles_centers(key_bin, min_area=min_area, max_area=max_area, min_circularity=min_circ)

    id_centers, q_centers = split_id_and_questions(centers, w, h)

    id_grid = build_grid(id_centers, rows=id_rows, cols=id_digits)
    q_grid = build_grid(q_centers, rows=num_q, cols=num_choices)

    dbg["centers_total"] = centers
    dbg["id_centers"] = id_centers
    dbg["q_centers"] = q_centers
    dbg["key_bin"] = key_bin

    if id_grid is None:
        raise ValueError("فشل تعلم شبكة الكود (ID). جرّب رفع DPI أو تعديل min_area/max_area/min_circularity.")
    if q_grid is None:
        raise ValueError("فشل تعلم شبكة الأسئلة. جرّب رفع DPI أو تعديل min_area/max_area/min_circularity.")

    template = LearnedTemplate(
        ref_bgr=key_bgr,
        ref_w=w,
        ref_h=h,
        id_grid=id_grid,
        q_grid=q_grid,
        num_q=num_q,
        num_choices=num_choices,
        id_rows=id_rows,
        id_digits=id_digits
    )
    return template, dbg


def extract_answer_key(template: LearnedTemplate,
                       min_fill: float,
                       ratio: float,
                       choices: List[str]) -> Tuple[Dict[int, str], pd.DataFrame]:
    """Read key answers directly from the answer key page using learned q_grid."""
    key_bin = preprocess_binary(template.ref_bgr)
    rows = template.q_grid.rows
    cols = template.q_grid.cols

    key = {}
    rows_dbg = []
    for r in range(rows):
        fills = []
        for c in range(cols):
            cx, cy = template.q_grid.centers[r, c]
            fills.append(fill_ratio(key_bin, int(cx), int(cy), win=18))
        ans, status = pick_choice(fills, choices, min_fill=min_fill, ratio=ratio)
        if status == "OK":
            key[r + 1] = ans
        rows_dbg.append([r + 1, ans, status] + [round(x, 3) for x in fills])

    df_dbg = pd.DataFrame(rows_dbg, columns=["Q", "Picked", "Status"] + choices)
    return key, df_dbg


def read_student_id(template: LearnedTemplate,
                    bin_img: np.ndarray,
                    min_fill: float,
                    ratio: float) -> Tuple[str, pd.DataFrame]:
    """Read 4 digits, each digit column selects row (0..9)."""
    id_rows = template.id_grid.rows
    id_cols = template.id_grid.cols

    cols_dbg = []
    digits = []
    for c in range(id_cols):
        fills = []
        for r in range(id_rows):
            cx, cy = template.id_grid.centers[r, c]
            fills.append(fill_ratio(bin_img, int(cx), int(cy), win=16))

        # Decide digit row
        idx = np.argsort(fills)[::-1]
        top_r = int(idx[0])
        top_fill = fills[top_r]
        second = fills[int(idx[1])] if len(idx) > 1 else 0.0

        if top_fill < min_fill:
            digits.append("X")
        elif second >= min_fill and (top_fill / (second + 1e-9)) < ratio:
            digits.append("X")
        else:
            digits.append(str(top_r))

        cols_dbg.append([c + 1] + [round(x, 3) for x in fills])

    df_dbg = pd.DataFrame(cols_dbg, columns=["DigitCol"] + [str(i) for i in range(id_rows)])
    return "".join(digits), df_dbg


def read_student_answers(template: LearnedTemplate,
                         bin_img: np.ndarray,
                         choices: List[str],
                         min_fill: float,
                         ratio: float) -> pd.DataFrame:
    rows = template.q_grid.rows
    cols = template.q_grid.cols

    out = []
    for r in range(rows):
        fills = []
        for c in range(cols):
            cx, cy = template.q_grid.centers[r, c]
            fills.append(fill_ratio(bin_img, int(cx), int(cy), win=18))

        ans, status = pick_choice(fills, choices, min_fill=min_fill, ratio=ratio)
        out.append([r + 1, ans, status] + [round(x, 3) for x in fills])

    return pd.DataFrame(out, columns=["Q", "Picked", "Status"] + choices)


# ==============================
# UI
# ==============================
def main():
    st.set_page_config(page_title="OMR Auto-Learn from Answer Key", layout="wide")
    st.title("✅ OMR يتعلّم القالب من الأنسر تلقائيًا (كود + أسئلة)")

    st.info("الفكرة: ترفع Answer Key مرة → البرنامج يكتشف شبكة الكود (10×4) + شبكة الأسئلة (10×4) تلقائيًا، ثم يصحح أوراق الطلاب.")

    col1, col2, col3 = st.columns(3)
    with col1:
        roster_file = st.file_uploader("📋 ملف الطلاب (Excel/CSV)", type=["xlsx", "xls", "csv"])
    with col2:
        key_file = st.file_uploader("🔑 Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    with col3:
        sheets_file = st.file_uploader("📚 أوراق الطلاب (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])

    st.markdown("---")
    st.subheader("إعدادات النموذج (حسب ورقتكم)")
    # According to your answer sheet: ID rows 0..9 => 10 rows, 4 digits columns, questions 1..10, choices A-D.
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

    st.subheader("إعدادات اكتشاف الفقاعات والتظليل")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        min_area = st.number_input("min_area", 20, 2000, 120, 10)
    with d2:
        max_area = st.number_input("max_area", 1000, 30000, 9000, 500)
    with d3:
        min_circ = st.slider("min_circularity", 0.30, 0.95, 0.55, 0.01)
    with d4:
        debug = st.checkbox("Debug (عرض الخطوات)", value=True)

    e1, e2 = st.columns(2)
    with e1:
        min_fill = st.slider("min_fill (حساسية التظليل)", 0.03, 0.35, 0.12, 0.01)
    with e2:
        double_ratio = st.slider("double_ratio (كشف التظليل المزدوج)", 1.10, 2.00, 1.35, 0.01)

    if not (roster_file and key_file and sheets_file):
        st.warning("ارفع الملفات الثلاثة حتى نبدأ.")
        return

    # Load roster
    try:
        roster = load_roster(roster_file)
        st.success(f"✅ تم تحميل قائمة الطلاب: {len(roster)}")
    except Exception as e:
        st.error(f"خطأ ملف الطلاب: {e}")
        return

    # Load key (first page)
    try:
        key_bytes = read_bytes(key_file)
        key_pages = load_pages(key_bytes, key_file.name, dpi=int(dpi))
        if not key_pages:
            st.error("فشل قراءة Answer Key")
            return
        key_bgr = pil_to_bgr(key_pages[0])
    except Exception as e:
        st.error(f"فشل فتح Answer Key: {e}")
        return

    # Learn template
    try:
        template, dbg = learn_template_from_answer_key(
            key_bgr=key_bgr,
            id_rows=int(id_rows),
            id_digits=int(id_digits),
            num_q=int(num_q),
            num_choices=int(num_choices),
            min_area=int(min_area),
            max_area=int(max_area),
            min_circ=float(min_circ),
        )
        st.success("✅ تم تعلم القالب تلقائيًا من صفحة الأنسر.")
    except Exception as e:
        st.error(f"❌ فشل تعلم القالب: {e}")
        return

    # Extract key answers
    answer_key, df_key_dbg = extract_answer_key(template, min_fill=float(min_fill), ratio=float(double_ratio), choices=choices)
    st.write("🔑 Answer Key المستخرج:")
    st.write(answer_key)

    if debug:
        st.markdown("---")
        st.subheader("Debug: Answer Key تعلم القالب")
        st.image(bgr_to_rgb(key_bgr), caption="Answer Key (Original)", width="stretch")
        st.image(dbg["key_bin"], caption="Answer Key (Binary)", width="stretch")

        # visualize centers
        vis = key_bgr.copy()
        for (x, y) in dbg["centers_total"]:
            cv2.circle(vis, (int(x), int(y)), 6, (255, 0, 0), 1)
        # ID centers red
        for (x, y) in dbg["id_centers"]:
            cv2.circle(vis, (int(x), int(y)), 7, (0, 0, 255), 2)
        # Q centers green
        for (x, y) in dbg["q_centers"]:
            cv2.circle(vis, (int(x), int(y)), 7, (0, 255, 0), 2)
        st.image(bgr_to_rgb(vis), caption="Detected Centers (Red=ID, Green=Questions)", width="stretch")

        # show grid points only
        vis2 = key_bgr.copy()
        for r in range(template.id_grid.rows):
            for c in range(template.id_grid.cols):
                x, y = template.id_grid.centers[r, c]
                cv2.circle(vis2, (int(x), int(y)), 9, (0, 0, 255), 2)
        for r in range(template.q_grid.rows):
            for c in range(template.q_grid.cols):
                x, y = template.q_grid.centers[r, c]
                cv2.circle(vis2, (int(x), int(y)), 9, (0, 255, 0), 2)
        st.image(bgr_to_rgb(vis2), caption="Learned Grid Points (Red=ID grid, Green=Q grid)", width="stretch")

        st.subheader("Debug: Key Answers fills table")
        st.dataframe(df_key_dbg, width="stretch", height=380)

    # Grade
    st.markdown("---")
    if st.button("🚀 ابدأ التصحيح", type="primary", width="stretch"):
        sheets_bytes = read_bytes(sheets_file)
        pages = load_pages(sheets_bytes, sheets_file.name, dpi=int(dpi))
        if not pages:
            st.error("فشل قراءة أوراق الطلاب")
            return

        results = []
        debug_samples = []

        prog = st.progress(0)
        for i, pil_page in enumerate(pages, start=1):
            page_bgr = pil_to_bgr(pil_page)
            aligned, ok, good_matches = orb_align(page_bgr, template.ref_bgr)
            bin_img = preprocess_binary(aligned)

            student_code, df_id_dbg = read_student_id(template, bin_img, min_fill=float(min_fill), ratio=float(double_ratio))
            student_name = roster.get(student_code, "غير موجود")

            df_ans = read_student_answers(template, bin_img, choices=choices, min_fill=float(min_fill), ratio=float(double_ratio))

            # score
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
                debug_samples.append((i, aligned, bin_img, df_id_dbg, df_ans))

            prog.progress(int(i / len(pages) * 100))

        df_res = pd.DataFrame(results)
        st.success("✅ اكتمل التصحيح")
        st.dataframe(df_res, width="stretch", height=420)

        # Export
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
            for page_no, aligned, bin_img, df_id_dbg, df_ans in debug_samples:
                st.subheader(f"صفحة {page_no}")
                st.image(bgr_to_rgb(aligned), caption="Aligned to Answer Key", width="stretch")
                st.image(bin_img, caption="Binary", width="stretch")
                st.subheader("ID fills table")
                st.dataframe(df_id_dbg, width="stretch")
                st.subheader("Answers fills table")
                st.dataframe(df_ans, width="stretch", height=360)


if __name__ == "__main__":
    main()
