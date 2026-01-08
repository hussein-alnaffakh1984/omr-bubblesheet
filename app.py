import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image


# =========================
# Utils: file read
# =========================
def read_uploaded_file_bytes(uploaded_file) -> bytes:
    if uploaded_file is None:
        return b""
    try:
        return uploaded_file.getbuffer().tobytes()
    except Exception:
        try:
            return uploaded_file.read()
        except Exception:
            return b""


# =========================
# Load pages
# =========================
def load_all_pages(file_bytes: bytes, filename: str, dpi: int = 250) -> List[Image.Image]:
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(p: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(p, cv2.COLOR_BGR2RGB)


# =========================
# Template model
# =========================
@dataclass
class BubbleGrid:
    # centers[r][c] = (x,y)
    centers: np.ndarray  # shape (rows, cols, 2)
    rows: int
    cols: int

@dataclass
class AutoTemplate:
    ref_w: int
    ref_h: int
    id_grid: BubbleGrid
    q_grid: BubbleGrid
    # number->name mapping
    num_questions: int
    num_choices: int
    id_rows: int
    id_digits: int


# =========================
# Image preprocess
# =========================
def preprocess_for_contours(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Invert binary: marks/edges -> white
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )
    bin_img = cv2.medianBlur(bin_img, 3)
    return bin_img


# =========================
# Detect bubbles (circles via contour circularity)
# =========================
def find_bubble_centers(bin_img: np.ndarray,
                        min_area: int = 80,
                        max_area: int = 5000,
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
        circ = 4 * np.pi * area / (peri * peri)
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


# =========================
# 1D clustering into k bins by sorting + splitting
# (robust for grid rows/cols)
# =========================
def cluster_1d(values: np.ndarray, k: int) -> np.ndarray:
    """
    values: shape (N,)
    returns labels 0..k-1 based on sorted split (equal-count bins)
    """
    idx = np.argsort(values)
    labels = np.zeros(len(values), dtype=np.int32)
    n = len(values)
    # equal count per bin
    for j in range(k):
        start = int(j * n / k)
        end = int((j + 1) * n / k)
        labels[idx[start:end]] = j
    return labels


def build_grid_from_centers(centers: np.ndarray, rows: int, cols: int) -> Optional[BubbleGrid]:
    """
    Given a set of bubble centers belonging to one grid,
    cluster into rows by y and cols by x, then compute average center per cell.
    """
    if centers.shape[0] < rows * cols * 0.7:
        return None

    xs = centers[:, 0].astype(np.float32)
    ys = centers[:, 1].astype(np.float32)

    rlab = cluster_1d(ys, rows)
    clab = cluster_1d(xs, cols)

    grid = np.zeros((rows, cols, 2), dtype=np.float32)
    counts = np.zeros((rows, cols), dtype=np.int32)

    for (x, y), r, c in zip(centers, rlab, clab):
        grid[r, c, 0] += x
        grid[r, c, 1] += y
        counts[r, c] += 1

    # replace empty cells by nearest non-empty (simple fallback)
    for r in range(rows):
        for c in range(cols):
            if counts[r, c] > 0:
                grid[r, c] /= counts[r, c]
            else:
                # fallback: use median of row/col
                row_pts = grid[r, counts[r] > 0] if np.any(counts[r] > 0) else None
                col_pts = grid[counts[:, c] > 0, c] if np.any(counts[:, c] > 0) else None
                if row_pts is not None and len(row_pts) > 0:
                    grid[r, c] = np.median(row_pts, axis=0)
                elif col_pts is not None and len(col_pts) > 0:
                    grid[r, c] = np.median(col_pts, axis=0)
                else:
                    grid[r, c] = (0, 0)

    return BubbleGrid(centers=grid, rows=rows, cols=cols)


# =========================
# Auto-split centers into two main grids (ID + Questions)
# Strategy: KMeans-like split by x coordinate (left vs right)
# =========================
def split_left_right(centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if centers.shape[0] == 0:
        return centers, centers
    xs = centers[:, 0]
    mid = np.median(xs)
    left = centers[xs < mid]
    right = centers[xs >= mid]
    return left, right


# =========================
# ORB Align (student page -> key reference)
# =========================
def orb_align_to_ref(student_bgr: np.ndarray, ref_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Returns warped student aligned to ref size using ORB+Homography.
    """
    h, w = ref_bgr.shape[:2]
    orb = cv2.ORB_create(4000)

    g1 = cv2.cvtColor(student_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)

    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return cv2.resize(student_bgr, (w, h)), False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 20:
        return cv2.resize(student_bgr, (w, h)), False

    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return cv2.resize(student_bgr, (w, h)), False

    warped = cv2.warpPerspective(student_bgr, H, (w, h))
    return warped, True


# =========================
# Fill measure at bubble center (small square window)
# =========================
def fill_ratio_at(bin_img: np.ndarray, cx: int, cy: int, win: int = 18) -> float:
    h, w = bin_img.shape[:2]
    x1 = max(0, cx - win)
    x2 = min(w, cx + win)
    y1 = max(0, cy - win)
    y2 = min(h, cy + win)
    patch = bin_img[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    # inner area only
    return float(np.sum(patch > 0)) / float(patch.size)


def pick_answer(fills: List[float], choices: List[str], min_fill: float, ratio: float) -> Tuple[str, str]:
    idx = np.argsort(fills)[::-1]
    top = int(idx[0])
    top_fill = fills[top]
    second = fills[int(idx[1])] if len(fills) > 1 else 0.0

    if top_fill < min_fill:
        return "?", "BLANK"
    if second >= min_fill and (top_fill / (second + 1e-9)) < ratio:
        return "!", "DOUBLE"
    return choices[top], "OK"


# =========================
# Build template from KEY
# =========================
def auto_build_template_from_key(key_bgr: np.ndarray,
                                 id_rows: int,
                                 id_digits: int,
                                 num_q: int,
                                 num_choices: int,
                                 debug: bool = True) -> AutoTemplate:
    ref_h, ref_w = key_bgr.shape[:2]
    bin_img = preprocess_for_contours(key_bgr)
    centers = find_bubble_centers(bin_img)

    if debug:
        st.write(f"عدد الفقاعات المكتشفة في الـKey: {len(centers)}")

    # Split by left/right
    left, right = split_left_right(centers)

    # Determine which side is Q and which side is ID by spread/shape:
    # Q grid عادة أكبر عددًا
    if len(left) >= len(right):
        q_centers = left
        id_centers = right
    else:
        q_centers = right
        id_centers = left

    id_grid = build_grid_from_centers(id_centers, rows=id_rows, cols=id_digits)
    q_grid = build_grid_from_centers(q_centers, rows=num_q, cols=num_choices)

    if id_grid is None or q_grid is None:
        raise ValueError("فشل بناء الشبكات تلقائيًا. (قد تكون الفقاعات غير واضحة أو الإعدادات تحتاج ضبط)")

    return AutoTemplate(
        ref_w=ref_w,
        ref_h=ref_h,
        id_grid=id_grid,
        q_grid=q_grid,
        num_questions=num_q,
        num_choices=num_choices,
        id_rows=id_rows,
        id_digits=id_digits
    )


# =========================
# Roster loader
# =========================
def load_roster(roster_file) -> Dict[str, str]:
    if roster_file.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(roster_file)
    else:
        df = pd.read_csv(roster_file)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "student_code" not in df.columns or "student_name" not in df.columns:
        raise ValueError("ملف الطلاب يجب أن يحتوي: student_code و student_name")

    codes = df["student_code"].astype(str).str.strip().str.replace(".0", "", regex=False)
    # pad to 4 digits (أو حسب id_digits في الواجهة)
    return dict(zip(codes, df["student_name"].astype(str).str.strip()))


# =========================
# Main app
# =========================
def main():
    st.set_page_config(page_title="Auto OMR (Key Learns Template)", layout="wide")
    st.title("✅ OMR ذكي داخل البرنامج: يتعلم القالب من Answer Key تلقائيًا")

    col1, col2, col3 = st.columns(3)
    with col1:
        roster_file = st.file_uploader("📋 ملف الطلاب (Excel/CSV)", type=["xlsx", "xls", "csv"])
    with col2:
        key_file = st.file_uploader("🔑 Answer Key (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"])
    with col3:
        sheets_file = st.file_uploader("📚 أوراق الطلاب (PDF) أو صور", type=["pdf", "png", "jpg", "jpeg"])

    st.markdown("---")
    st.subheader("إعدادات (مرة واحدة فقط)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dpi = st.slider("DPI", 150, 350, 250, 10)
    with c2:
        id_digits = st.number_input("خانات كود الطالب", 1, 12, 4, 1)
    with c3:
        id_rows = st.number_input("صفوف الكود (0-9)", 10, 15, 10, 1)
    with c4:
        num_q = st.number_input("عدد الأسئلة", 1, 200, 10, 1)

    num_choices = st.selectbox("عدد الخيارات", [4, 5, 6], index=0)
    choices = list("ABCDEF"[:num_choices])

    min_fill = st.slider("min_fill (حساسية التظليل)", 0.03, 0.35, 0.12, 0.01)
    double_ratio = st.slider("double_ratio", 1.10, 2.00, 1.35, 0.01)
    debug = st.checkbox("عرض Debug", value=True)

    if not (roster_file and key_file and sheets_file):
        st.info("ارفع الملفات الثلاثة ثم نبدأ.")
        return

    # Load roster
    try:
        roster = load_roster(roster_file)
        st.success(f"✅ تم تحميل قائمة الطلاب: {len(roster)}")
    except Exception as e:
        st.error(f"خطأ ملف الطلاب: {e}")
        return

    # Load key page
    key_bytes = read_uploaded_file_bytes(key_file)
    key_pages = load_all_pages(key_bytes, key_file.name, dpi=dpi)
    if not key_pages:
        st.error("فشل قراءة Answer Key")
        return

    key_bgr = pil_to_bgr(key_pages[0])

    # Build template automatically
    try:
        template = auto_build_template_from_key(
            key_bgr,
            id_rows=int(id_rows),
            id_digits=int(id_digits),
            num_q=int(num_q),
            num_choices=int(num_choices),
            debug=debug
        )
        st.success("✅ تم تعلم القالب تلقائيًا من الـAnswer Key")
    except Exception as e:
        st.error(f"❌ فشل تعلم القالب: {e}")
        st.info("إذا الفشل بسبب ضعف الفقاعات: ارفع DPI أو ارفع min_fill قليلاً، أو أرسل صورة أوضح للـKey.")
        return

    # Extract answer key from key itself using q_grid
    key_bin = preprocess_for_contours(key_bgr)
    answer_key = {}
    dbg_key_table = []
    for r in range(template.q_grid.rows):
        fills = []
        for c in range(template.q_grid.cols):
            cx, cy = template.q_grid.centers[r, c]
            fills.append(fill_ratio_at(key_bin, int(cx), int(cy), win=18))
        ans, status = pick_answer(fills, choices, min_fill=min_fill, ratio=double_ratio)
        if status == "OK":
            answer_key[r + 1] = ans
        dbg_key_table.append([r + 1, ans, status] + [round(x, 3) for x in fills])

    st.write("🔑 Answer Key المستخرج:")
    st.write(answer_key)

    if debug:
        st.subheader("Debug Key: bubble centers")
        vis = key_bgr.copy()
        # draw ID centers
        for r in range(template.id_grid.rows):
            for c in range(template.id_grid.cols):
                x, y = template.id_grid.centers[r, c]
                cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), 2)
        # draw Q centers
        for r in range(template.q_grid.rows):
            for c in range(template.q_grid.cols):
                x, y = template.q_grid.centers[r, c]
                cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0), 2)

        st.image(bgr_to_rgb(vis), caption="Key with detected centers (Red=ID, Green=Questions)", use_container_width=True)

        df_dbg_key = pd.DataFrame(
            dbg_key_table,
            columns=["Q", "picked", "status"] + choices
        )
        st.dataframe(df_dbg_key, use_container_width=True)

    # Grade students
    if st.button("🚀 ابدأ التصحيح", type="primary", use_container_width=True):
        sheets_bytes = read_uploaded_file_bytes(sheets_file)
        pages = load_all_pages(sheets_bytes, sheets_file.name, dpi=dpi)
        if not pages:
            st.error("فشل قراءة أوراق الطلاب")
            return

        results = []
        sample_debug = []

        prog = st.progress(0)
        for i, pil_page in enumerate(pages, 1):
            stud_bgr = pil_to_bgr(pil_page)
            aligned, ok = orb_align_to_ref(stud_bgr, key_bgr)
            bin_img = preprocess_for_contours(aligned)

            # Read student ID
            digits = []
            id_debug_cols = []
            for c in range(template.id_grid.cols):
                fills = []
                for r in range(template.id_grid.rows):
                    cx, cy = template.id_grid.centers[r, c]
                    fills.append(fill_ratio_at(bin_img, int(cx), int(cy), win=16))
                # choose row index as digit (0..9)
                idx = np.argsort(fills)[::-1]
                top_r = int(idx[0])
                top_fill = fills[top_r]
                second = fills[int(idx[1])] if len(idx) > 1 else 0.0
                if top_fill < min_fill:
                    digits.append("X")
                elif second >= min_fill and (top_fill / (second + 1e-9)) < double_ratio:
                    digits.append("X")
                else:
                    digits.append(str(top_r))
                id_debug_cols.append([c + 1] + [round(x, 3) for x in fills])

            student_code = "".join(digits)
            student_name = roster.get(student_code, "غير موجود")

            # Read answers
            correct = 0
            per_q_dbg = []
            for q in range(1, template.num_questions + 1):
                fills = []
                for c in range(template.num_choices):
                    cx, cy = template.q_grid.centers[q - 1, c]
                    fills.append(fill_ratio_at(bin_img, int(cx), int(cy), win=18))
                ans, status = pick_answer(fills, choices, min_fill=min_fill, ratio=double_ratio)

                key_ans = answer_key.get(q, None)
                is_ok = (status == "OK" and key_ans is not None and ans == key_ans)
                correct += int(is_ok)

                per_q_dbg.append([q, ans, status, key_ans, is_ok] + [round(x, 3) for x in fills])

            total = len(answer_key)
            pct = (correct / total * 100) if total else 0.0
            passed = "ناجح" if pct >= 50 else "راسب"

            results.append({
                "page": i,
                "aligned_ok": ok,
                "student_code": student_code,
                "student_name": student_name,
                "score": correct,
                "total": total,
                "percentage": round(pct, 2),
                "status": passed
            })

            if debug and len(sample_debug) < 2:
                sample_debug.append((i, aligned, bin_img, id_debug_cols, per_q_dbg))

            prog.progress(int(i / len(pages) * 100))

        df = pd.DataFrame(results)
        st.success("✅ اكتمل التصحيح")
        st.dataframe(df, use_container_width=True, height=420)

        # export
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        st.download_button(
            "⬇️ تحميل النتائج Excel",
            data=buf.getvalue(),
            file_name="results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        # debug samples
        if debug and sample_debug:
            st.markdown("---")
            st.header("Debug Samples")
            for page_no, aligned, bin_img, id_cols, per_q in sample_debug:
                st.subheader(f"Page {page_no}")
                st.image(bgr_to_rgb(aligned), caption="Aligned to Key", use_container_width=True)
                st.image(bin_img, caption="Binary", clamp=True, use_container_width=True)

                st.write("ID fills (كل عمود = 10 قيم):")
                df_id = pd.DataFrame(id_cols, columns=["digit_col"] + [f"r{i}" for i in range(template.id_rows)])
                st.dataframe(df_id, use_container_width=True)

                st.write("Answers debug:")
                df_q = pd.DataFrame(per_q, columns=["Q", "picked", "status", "key", "correct"] + choices)
                st.dataframe(df_q, use_container_width=True)

if __name__ == "__main__":
    main()
