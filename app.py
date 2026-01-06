# app.py  (FULL REPLACEMENT - Streamlit Cloud Ready)
# --------------------------------------------------
# ✅ Reads student code from OMR grid
# ✅ Grades answers from Answer Key
# ✅ Outputs Excel: sheet_index, student_code, student_name, score
# ✅ Works in batches for large PDFs (avoid RAM crash on Streamlit Cloud)
# ✅ Auto-scales ROIs for different DPI/scan sizes
# ✅ Debug mode to preview ROIs
# ✅ Option: treat BLANK/DOUBLE as wrong (default)

import io
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterator

import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image

# --- OpenCV (required for this version)
try:
    import cv2
except Exception as e:
    cv2 = None

# =========================
# 1) Template Configuration (Based on your uploaded sheet)
# =========================
@dataclass
class TemplateConfig:
    # ROI for student code grid (x, y, w, h) on reference image size
    id_roi: Tuple[int, int, int, int]

    # student code columns (digits count)
    id_digits: int

    # rows for digits (0-9)
    id_rows: int

    # Question blocks: (x, y, w, h, start_q, end_q)
    q_blocks: List[Tuple[int, int, int, int, int, int]]

    # number of questions vertically in each block
    block_rows: int


# Reference size observed from your sample (DPI ~120)
REF_W, REF_H = 991, 1420

# ✅ These coordinates are tuned for your Hematology sheet sample
BASE_CFG = TemplateConfig(
    id_roi=(600, 140, 320, 420),
    id_digits=4,
    id_rows=10,
    q_blocks=[
        (120, 520, 260, 860, 1, 20),
        (440, 520, 260, 860, 21, 40),
        (760, 520, 220, 860, 41, 60),
    ],
    block_rows=20
)


def scale_box(box: Tuple[int, int, int, int], sx: float, sy: float) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))


def cfg_for_image(img_w: int, img_h: int) -> TemplateConfig:
    """Scale reference config to current image size."""
    sx, sy = img_w / REF_W, img_h / REF_H
    scaled_id = scale_box(BASE_CFG.id_roi, sx, sy)
    scaled_blocks = []
    for (x, y, w, h, qs, qe) in BASE_CFG.q_blocks:
        x2, y2, w2, h2 = scale_box((x, y, w, h), sx, sy)
        scaled_blocks.append((x2, y2, w2, h2, qs, qe))
    return TemplateConfig(
        id_roi=scaled_id,
        id_digits=BASE_CFG.id_digits,
        id_rows=BASE_CFG.id_rows,
        q_blocks=scaled_blocks,
        block_rows=BASE_CFG.block_rows
    )


# =========================
# 2) Helpers
# =========================
def parse_ranges(txt: str) -> List[Tuple[int, int]]:
    """Parse ranges like: 1-40, 45-60, 70"""
    if not txt or not txt.strip():
        return []
    out = []
    for part in txt.split(","):
        p = part.strip()
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", p)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.append((min(a, b), max(a, b)))
        elif p.isdigit():
            x = int(p)
            out.append((x, x))
    return out


def in_ranges(q: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(a <= q <= b for a, b in ranges)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """Return binary image (filled marks => white pixels in THRESH_BINARY_INV)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )
    return thr


def score_cell(bin_img: np.ndarray) -> int:
    return int(np.sum(bin_img > 0))


def pick_one(scores: List[Tuple[str, int]], min_fill: int, min_ratio: float) -> Tuple[str, str]:
    """
    Returns:
      ("A","OK") or ("?","BLANK") or ("!","DOUBLE")
    """
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0

    if top_s < min_fill:
        return "?", "BLANK"
    if second_s > 0 and (top_s / (second_s + 1e-6)) < min_ratio:
        return "!", "DOUBLE"
    return top_c, "OK"


# =========================
# 3) OMR Readers
# =========================
def read_student_code(thr: np.ndarray, cfg: TemplateConfig,
                      min_fill: int = 180, min_ratio: float = 1.25) -> str:
    """
    Reads student code from a vertical grid:
      rows = 0..9
      columns = digits count
    """
    x, y, w, h = cfg.id_roi
    roi = thr[y:y + h, x:x + w]

    rows, cols = cfg.id_rows, cfg.id_digits
    ch, cw = h // rows, w // cols

    digits = []
    for c in range(cols):
        col_scores = []
        for r in range(rows):
            cell = roi[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            col_scores.append((str(r), score_cell(cell)))

        d, status = pick_one(col_scores, min_fill=min_fill, min_ratio=min_ratio)
        digits.append("" if d in ["?", "!"] else d)

    code = "".join(digits).strip()
    # Force 4 digits as your sheet:
    if code:
        code = code.zfill(cfg.id_digits)
    return code


def read_answers(thr: np.ndarray, cfg: TemplateConfig, choices: int,
                 min_fill: int = 140, min_ratio: float = 1.25) -> Dict[int, Tuple[str, str]]:
    """
    Reads answers from question blocks.
    Returns dict: {q: (answer_letter, status)}
    status in {"OK","BLANK","DOUBLE"}
    """
    letters = "ABCDE"[:choices]
    out: Dict[int, Tuple[str, str]] = {}

    for (x, y, w, h, qs, qe) in cfg.q_blocks:
        roi = thr[y:y + h, x:x + w]
        rows = cfg.block_rows
        rh, cw = h // rows, w // choices

        q = qs
        for r in range(rows):
            if q > qe:
                break

            scores = []
            for c in range(choices):
                cell = roi[r * rh:(r + 1) * rh, c * cw:(c + 1) * cw]
                scores.append((letters[c], score_cell(cell)))

            a, stt = pick_one(scores, min_fill=min_fill, min_ratio=min_ratio)
            out[q] = (a, stt)
            q += 1

    return out


# =========================
# 4) PDF Handling (Batch to avoid RAM)
# =========================
def load_key_image(file_bytes: bytes, filename: str, dpi: int) -> Image.Image:
    """Answer key is single page PDF or image."""
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi, first_page=1, last_page=1)
        return pages[0]
    return Image.open(io.BytesIO(file_bytes))


def iter_pdf_pages(pdf_bytes: bytes, dpi: int, start_page: int, end_page: int) -> Iterator[Image.Image]:
    """Convert a range of PDF pages into PIL images."""
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=start_page, last_page=end_page)
    for p in pages:
        yield p


# =========================
# 5) Streamlit UI
# =========================
st.set_page_config(page_title="OMR Bubble Sheet", layout="wide")
st.title("تصحيح ببل شيت – Hematology (Cloud)")

if cv2 is None:
    st.error("OpenCV (cv2) غير متاح على السيرفر. تأكد من requirements.txt و runtime.txt (python-3.11).")
    st.stop()

with st.expander("⚙️ إعدادات الامتحان", expanded=True):
    choices = st.radio("عدد الخيارات لكل سؤال", [4, 5], horizontal=True)

    theory_txt = st.text_input("نطاق النظري (مثال: 1-60) — إذا تركته فارغ سيصحح كل الأسئلة")
    practical_txt = st.text_input("نطاق العملي (اختياري)")

    strict = st.checkbox("Strict: BLANK/DOUBLE تعتبر خطأ (موصى به)", value=True)

    dpi = st.slider("DPI لتحويل PDF (أقل = أخف وأسرع)", 80, 200, 120, 10)

    st.markdown("### ✅ Batch (لتفادي 200MB/RAM)")
    start_page = st.number_input("Start page", min_value=1, value=1, step=1)
    end_page = st.number_input("End page", min_value=1, value=50, step=1)

debug = st.checkbox("Debug: عرض مناطق القص (ROIs)", value=False)

theory_ranges = parse_ranges(theory_txt)
practical_ranges = parse_ranges(practical_txt)

st.subheader("1) ملف الطلاب (Roster)")
roster_file = st.file_uploader("Excel/CSV يحتوي: student_code, student_name", type=["xlsx", "xls", "csv"])

roster: Dict[str, str] = {}
if roster_file:
    fn = roster_file.name.lower()
    df = pd.read_csv(roster_file) if fn.endswith(".csv") else pd.read_excel(roster_file)
    df.columns = [c.strip().lower() for c in df.columns]

    if "student_code" not in df.columns or "student_name" not in df.columns:
        st.error("ملف roster لازم يحتوي عمودين بالضبط: student_code و student_name")
        st.stop()

    df["student_code"] = df["student_code"].astype(str).str.strip()
    df["student_name"] = df["student_name"].astype(str).str.strip()

    # Ensure codes are 4 digits (your sheet)
    df["student_code"] = df["student_code"].apply(lambda x: x.zfill(BASE_CFG.id_digits))

    roster = dict(zip(df["student_code"], df["student_name"]))
    st.success(f"تم تحميل {len(roster)} طالب")

st.subheader("2) Answer Key")
key_file = st.file_uploader("PDF صفحة واحدة أو صورة", type=["pdf", "png", "jpg", "jpeg"])

st.subheader("3) أوراق الطلاب")
sheets_file = st.file_uploader("PDF متعدد الصفحات أو صورة", type=["pdf", "png", "jpg", "jpeg"])


def compute_score(key_ans: Dict[int, Tuple[str, str]],
                  stu_ans: Dict[int, Tuple[str, str]],
                  theory_ranges: List[Tuple[int, int]],
                  practical_ranges: List[Tuple[int, int]]) -> int:
    score = 0

    # If user didn't specify any range, grade everything
    grade_all = (not theory_ranges and not practical_ranges)

    for q, (ka, kst) in key_ans.items():
        sa, sst = stu_ans.get(q, ("?", "BLANK"))

        # handle strictness: treat BLANK/DOUBLE as wrong always
        if sa in ["?", "!"]:
            continue

        if grade_all:
            if sa == ka:
                score += 1
            continue

        if theory_ranges and in_ranges(q, theory_ranges) and sa == ka:
            score += 1
        if practical_ranges and in_ranges(q, practical_ranges) and sa == ka:
            score += 1

    return score


if st.button("🚀 ابدأ التصحيح (Batch)"):
    if not roster:
        st.error("ارفع ملف roster أولاً")
        st.stop()
    if not key_file or not sheets_file:
        st.error("ارفع Answer Key وملف أوراق الطلاب")
        st.stop()
    if end_page < start_page:
        st.error("End page لازم يكون أكبر أو يساوي Start page")
        st.stop()

    # --- Prepare Answer Key
    key_bytes = key_file.getvalue()
    key_img = load_key_image(key_bytes, key_file.name, dpi=dpi)

    key_bgr = pil_to_bgr(key_img)
    key_thr = preprocess(key_bgr)
    Hk, Wk = key_thr.shape[:2]
    cfgk = cfg_for_image(Wk, Hk)

    if debug:
        x, y, w, h = cfgk.id_roi
        st.image(key_thr[y:y + h, x:x + w], caption="DEBUG - Answer Key: ID ROI", clamp=True)
        for i, (x, y, w, h, qs, qe) in enumerate(cfgk.q_blocks, 1):
            st.image(key_thr[y:y + h, x:x + w], caption=f"DEBUG - Answer Key: Block {i} ({qs}-{qe})", clamp=True)

    key_ans = read_answers(key_thr, cfgk, choices=choices)

    # --- Student pages
    sf_name = sheets_file.name.lower()
    sf_bytes = sheets_file.getvalue()

    results = []
    total = int(end_page - start_page + 1)
    prog = st.progress(0)
    status_box = st.empty()

    if sf_name.endswith(".pdf"):
        pages_iter = iter_pdf_pages(sf_bytes, dpi=dpi, start_page=int(start_page), end_page=int(end_page))
    else:
        pages_iter = [Image.open(io.BytesIO(sf_bytes))]

    for idx, pg in enumerate(pages_iter, 1):
        sheet_index = int(start_page) + idx - 1

        bgr = pil_to_bgr(pg)
        thr = preprocess(bgr)
        H, W = thr.shape[:2]
        cfg = cfg_for_image(W, H)

        if debug and idx == 1:
            x, y, w, h = cfg.id_roi
            st.image(thr[y:y + h, x:x + w], caption="DEBUG - Student: ID ROI (first page only)", clamp=True)
            for i, (x, y, w, h, qs, qe) in enumerate(cfg.q_blocks, 1):
                st.image(thr[y:y + h, x:x + w], caption=f"DEBUG - Student: Block {i} ({qs}-{qe})", clamp=True)

        code = read_student_code(thr, cfg)
        name = roster.get(code, "")

        stu_ans = read_answers(thr, cfg, choices=choices)
        score = compute_score(key_ans, stu_ans, theory_ranges, practical_ranges)

        results.append({
            "sheet_index": sheet_index,
            "student_code": code,
            "student_name": name,
            "score": score
        })

        prog.progress(int(idx / total * 100))
        status_box.info(f"Processing page {sheet_index}  |  Code: {code}  |  Score: {score}")

    out = pd.DataFrame(results)
    buf = io.BytesIO()
    out.to_excel(buf, index=False)

    st.success("✅ تم إنشاء ملف Excel بنجاح")
    st.download_button(
        "⬇️ تحميل Excel",
        data=buf.getvalue(),
        file_name=f"results_{start_page}_{end_page}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
