"""
OMR PRO (Scanner-PDF) — Zero Settings / Zero ROI
✅ Student Code: bubbles-only, auto-detect 4x10 grid (NO OCR, NO AI, NO ROI)
✅ Works on clean scanner PDFs (consistent layout)
✅ Review only when truly ambiguous/faint/missing
✅ Results table + duplicates + export Excel

pip install streamlit opencv-python numpy pandas pdf2image pillow openpyxl
Linux: install poppler for pdf2image
"""

import io, gc
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
from datetime import datetime


# =========================
# Fixed production constants (NO USER SETTINGS)
# =========================
DPI = 260                 # scanner pdf sweet spot
TOP_REGION_RATIO = 0.50   # code grid is usually in upper half
MIN_CIRCULARITY = 0.55
MIN_AREA = 60
R_MIN = 6
R_MAX = 45

# ink / ambiguity thresholds (fixed)
MIN_INK = 0.030
MIN_MARGIN = 0.010

# allowed code range (optional safety)
CODE_MIN = 1000
CODE_MAX = 1999


# =========================
# Helpers
# =========================
def read_bytes(f):
    if not f:
        return b""
    try:
        return f.getbuffer().tobytes()
    except Exception:
        try:
            return f.read()
        except Exception:
            return b""


def load_pages(file_bytes: bytes, filename: str, dpi: int = DPI) -> List[Image.Image]:
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(
            file_bytes,
            dpi=dpi,
            fmt="jpeg",
            jpegopt={"quality": 90, "optimize": True},
        )
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# =========================
# 1D KMeans clustering
# =========================
def kmeans_1d(vals: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    vals = vals.astype(np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.01)
    _, labels, centers = cv2.kmeans(vals, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = centers.flatten()
    order = np.argsort(centers)

    remap = np.zeros_like(order)
    for new, old in enumerate(order):
        remap[old] = new

    labels_sorted = np.array([remap[int(l)] for l in labels.flatten()], dtype=int)
    centers_sorted = centers[order]
    return labels_sorted, centers_sorted


# =========================
# Ink score inside bubble
# =========================
def ink_score_in_circle(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    r2 = max(8, int(r * 0.85))
    y1, y2 = max(0, cy - r2), min(gray.shape[0], cy + r2)
    x1, x2 = max(0, cx - r2), min(gray.shape[1], cx + r2)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0

    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    patch = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX)

    th = cv2.adaptiveThreshold(
        patch, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 3
    )

    h, w = th.shape
    mask = np.zeros_like(th)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * 0.38), 255, -1)

    ink = cv2.countNonZero(cv2.bitwise_and(th, mask))
    area = cv2.countNonZero(mask) + 1e-6
    return float(ink) / float(area)


# =========================
# Auto-detect 4x10 code grid & read code
# =========================
@dataclass
class CodeResult:
    code: Optional[str]
    ok: bool
    reason: str
    row_scores: List[float]
    row_margins: List[float]
    debug: Dict


def read_code_auto(page_bgr: np.ndarray) -> CodeResult:
    dbg = {}
    H, W = page_bgr.shape[:2]

    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray_blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        peri = cv2.arcLength(c, True) + 1e-6
        circ = 4 * np.pi * area / (peri * peri)
        if circ < MIN_CIRCULARITY:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < R_MIN or r > R_MAX:
            continue
        circles.append((float(x), float(y), float(r)))

    dbg["circles_raw"] = len(circles)
    if len(circles) < 60:
        return CodeResult(None, False, "REVIEW: not enough bubbles detected", [], [], dbg)

    circles = np.array(circles, dtype=np.float32)
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]

    # keep dominant bubble size band
    r_med = float(np.median(rs))
    keep = (rs > r_med * 0.70) & (rs < r_med * 1.40)
    circles = circles[keep]
    dbg["circles_size_filtered"] = int(len(circles))
    if len(circles) < 50:
        return CodeResult(None, False, "REVIEW: bubble size filtering failed", [], [], dbg)

    # focus on top half (where code grid typically is)
    top = circles[circles[:, 1] < TOP_REGION_RATIO * H]
    dbg["circles_top"] = int(len(top))
    if len(top) < 40:
        top = circles  # fallback if layout differs

    pts = top
    x = pts[:, 0]
    y = pts[:, 1]

    # cluster y into 4 rows
    try:
        row_labels, row_centers = kmeans_1d(y, 4)
    except Exception:
        return CodeResult(None, False, "REVIEW: row clustering failed", [], [], dbg)

    # remove y-outliers
    ydist = np.abs(y - row_centers[row_labels])
    thr = np.percentile(ydist, 70) * 1.8 + 1e-6
    good = ydist < thr

    pts2 = pts[good]
    row_labels2 = row_labels[good]
    dbg["circles_after_row_clean"] = int(len(pts2))
    if len(pts2) < 35:
        return CodeResult(None, False, "REVIEW: row grid unstable", [], [], dbg)

    # cluster x into 10 columns
    try:
        col_labels, col_centers = kmeans_1d(pts2[:, 0], 10)
    except Exception:
        return CodeResult(None, False, "REVIEW: column clustering failed", [], [], dbg)

    # build cell -> best candidate bubble by distance to (row_center, col_center)
    grid = [[None for _ in range(10)] for _ in range(4)]
    for (cx, cy, r), rr, cc in zip(pts2, row_labels2, col_labels):
        dx = abs(cx - col_centers[cc])
        dy = abs(cy - row_centers[rr])
        d = dx + dy
        if grid[rr][cc] is None or d < grid[rr][cc][0]:
            grid[rr][cc] = (d, int(cx), int(cy), int(r))

    missing = sum(1 for rr in range(4) for cc in range(10) if grid[rr][cc] is None)
    dbg["grid_missing"] = int(missing)
    if missing > 8:
        return CodeResult(None, False, "REVIEW: code grid not found", [], [], dbg)

    # read darkest per row
    row_scores, row_margins = [], []
    digits = []

    for rr in range(4):
        scores = np.zeros((10,), dtype=np.float32)
        for cc in range(10):
            cell = grid[rr][cc]
            if cell is None:
                scores[cc] = 0.0
                continue
            _, cx, cy, r = cell
            scores[cc] = ink_score_in_circle(gray, cx, cy, r)

        best = int(np.argmax(scores))
        best_sc = float(scores[best])
        second_sc = float(np.partition(scores, -2)[-2])
        margin = best_sc - second_sc

        row_scores.append(best_sc)
        row_margins.append(margin)

        if best_sc < MIN_INK:
            return CodeResult(None, False, f"REVIEW: row {rr+1} too faint", row_scores, row_margins, dbg)
        if margin < MIN_MARGIN:
            return CodeResult(None, False, f"REVIEW: row {rr+1} ambiguous", row_scores, row_margins, dbg)

        digits.append(best)

    code = "".join(map(str, digits))
    if not code.isdigit():
        return CodeResult(None, False, "REVIEW: invalid code", row_scores, row_margins, dbg)

    code_int = int(code)
    if not (CODE_MIN <= code_int <= CODE_MAX):
        # if your codes are different, remove this range check
        return CodeResult(None, False, f"REVIEW: code out of range ({code})", row_scores, row_margins, dbg)

    return CodeResult(code, True, "OK", row_scores, row_margins, dbg)


# =========================
# Main App (Code-only)
# =========================
def main():
    st.set_page_config(page_title="OMR — Code from Bubbles (No Settings)", layout="wide")
    st.title("✅ OMR — استخراج كود الطالب من الفقاعات فقط (بدون أي إعدادات)")

    if "codes" not in st.session_state: st.session_state.codes = []
    if "review" not in st.session_state: st.session_state.review = []

    pdf = st.file_uploader("Upload Scanner PDF", type=["pdf"])

    col1, col2 = st.columns(2)
    with col1:
        batch = st.slider("Batch (للسرعة)", 5, 30, 10)
    with col2:
        show_debug = st.checkbox("Show debug for review pages", value=False)

    if pdf and st.button("🚀 Extract Codes", type="primary"):
        st.session_state.codes = []
        st.session_state.review = []

        b = read_bytes(pdf)
        pages = load_pages(b, pdf.name, dpi=DPI)

        prog = st.progress(0)
        status = st.empty()

        # process all pages
        for i, p in enumerate(pages):
            status.text(f"Page {i+1}/{len(pages)}")
            prog.progress((i+1)/len(pages))

            bgr = pil_to_bgr(p)
            res = read_code_auto(bgr)

            if res.ok:
                st.session_state.codes.append({"Page": i+1, "Code": res.code})
            else:
                row = {"Page": i+1, "Code": "REVIEW", "Reason": res.reason}
                if show_debug:
                    row["row_scores"] = res.row_scores
                    row["row_margins"] = res.row_margins
                    row["dbg"] = res.debug
                st.session_state.review.append(row)

            del bgr
            if (i % batch) == 0:
                gc.collect()

        gc.collect()
        st.success("✅ Done")

    # Results
    if st.session_state.codes:
        df = pd.DataFrame(st.session_state.codes)
        st.subheader("📌 Extracted Codes")
        st.dataframe(df, use_container_width=True)

        # duplicates
        dup = df.groupby("Code").size().reset_index(name="Count")
        dup = dup[dup["Count"] > 1]
        st.subheader("🔁 Duplicates")
        if len(dup):
            st.error(f"⚠️ Found {len(dup)} duplicate codes")
            st.dataframe(dup, use_container_width=True)
        else:
            st.success("✅ No duplicates")

        # export
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="Codes")
            if st.session_state.review:
                pd.DataFrame(st.session_state.review).to_excel(w, index=False, sheet_name="Review")
            if len(dup):
                dup.to_excel(w, index=False, sheet_name="Duplicates")

        st.download_button(
            "⬇️ Download Excel (Codes + Review + Duplicates)",
            out.getvalue(),
            f"codes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if st.session_state.review:
        st.subheader("🟨 Review Pages")
        rdf = pd.DataFrame(st.session_state.review)
        st.dataframe(rdf, use_container_width=True)


if __name__ == "__main__":
    main()
