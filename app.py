import io
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterator, Optional

import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image

# حاول نستورد cv2، وإذا فشل/سبب مشكلة نطلع رسالة واضحة
try:
    import cv2
    CV2_OK = True
except Exception as e:
    CV2_OK = False
    cv2 = None

# =========================
# Template Configuration
# =========================
@dataclass
class TemplateConfig:
    id_roi: Tuple[int,int,int,int] = (1200, 150, 600, 650)
    id_digits: int = 6
    id_rows: int = 10

    q_blocks: List[Tuple[int,int,int,int,int,int]] = None
    block_rows: int = 20

def default_config() -> TemplateConfig:
    cfg = TemplateConfig()
    cfg.q_blocks = [
        (150, 520, 550, 1900, 1, 20),
        (760, 520, 550, 1900, 21, 40),
        (1370, 520, 550, 1900, 41, 60),
    ]
    return cfg

CFG = default_config()

# =========================
# Utils
# =========================
def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )
    return thr

def score_cell(bin_img: np.ndarray) -> int:
    return int(np.sum(bin_img > 0))

def pick_one(scores, min_fill, min_ratio):
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0
    if top_s < min_fill:
        return "?", "BLANK"
    if second_s > 0 and (top_s / (second_s + 1e-6)) < min_ratio:
        return "!", "DOUBLE"
    return top_c, "OK"

def parse_ranges(txt: str) -> List[Tuple[int,int]]:
    if not txt.strip():
        return []
    out = []
    for part in txt.split(","):
        p = part.strip()
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", p)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.append((min(a,b), max(a,b)))
        elif p.isdigit():
            x = int(p)
            out.append((x,x))
    return out

def in_ranges(q: int, ranges: List[Tuple[int,int]]) -> bool:
    return any(a <= q <= b for a,b in ranges)

# =========================
# PDF loading (page-by-page)
# =========================
def iter_pdf_pages(pdf_bytes: bytes, dpi: int, start: int, end: Optional[int]) -> Iterator[Image.Image]:
    """
    يُرجع صفحات PDF واحدة واحدة لتقليل RAM.
    start/end (1-indexed). end=None يعني للنهاية.
    """
    # pdf2image تحتاج تحديد first_page/last_page لتحويل جزء
    # سنمشي batch-by-batch
    # ملاحظة: لا نعرف عدد الصفحات بسهولة بدون pdfinfo،
    # لذلك إذا end=None، المستخدم يحدد end يدويًا (نقترح دفعات).
    if end is None:
        raise ValueError("لازم تحدد نهاية الدفعة (end page) لتجنب استهلاك الذاكرة على Cloud.")
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=start, last_page=end)
    for p in pages:
        yield p

def load_single_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes))

# =========================
# OMR Readers
# =========================
def read_student_code(thr, cfg, min_fill=250, min_ratio=1.25):
    x,y,w,h = cfg.id_roi
    roi = thr[y:y+h, x:x+w]
    rows, cols = cfg.id_rows, cfg.id_digits
    ch, cw = h//rows, w//cols
    digits = []
    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            scores.append((str(r), score_cell(cell)))
        d, _ = pick_one(scores, min_fill, min_ratio)
        digits.append("" if d in ["?","!"] else d)
    return "".join(digits)

def read_answers(thr, cfg, choices, min_fill=180, min_ratio=1.25):
    letters = "ABCDE"[:choices]
    out = {}
    for (x,y,w,h,qs,qe) in cfg.q_blocks:
        roi = thr[y:y+h, x:x+w]
        rows = cfg.block_rows
        rh, cw = h//rows, w//choices
        q = qs
        for r in range(rows):
            if q > qe:
                break
            scores = []
            for c in range(choices):
                cell = roi[r*rh:(r+1)*rh, c*cw:(c+1)*cw]
                scores.append((letters[c], score_cell(cell)))
            a, st = pick_one(scores, min_fill, min_ratio)
            out[q] = (a, st)
            q += 1
    return out

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="OMR Bubble Sheet", layout="wide")
st.title("تصحيح ببل شيت – Streamlit Cloud (Excel فقط)")

if not CV2_OK:
    st.error("مكتبة OpenCV (cv2) لم تعمل على السيرفر. جرّب تثبيت Python 3.11 عبر runtime.txt أو استخدم نسخة بدون cv2.")
    st.stop()

with st.expander("⚙️ إعدادات الامتحان", expanded=True):
    choices = st.radio("عدد الخيارات", [4,5], horizontal=True)
    theory_txt = st.text_input("نطاق النظري (مثال: 1-40)")
    practical_txt = st.text_input("نطاق العملي (اختياري)")
    dpi = st.slider("DPI لتحويل PDF (أقل = أسرع وأخف)", 80, 200, 120, 10)

    # دفعات لمنع 200MB/ذاكرة
    st.markdown("**تصحيح على دفعات (Batch):**")
    start_page = st.number_input("Start page", min_value=1, value=1, step=1)
    end_page = st.number_input("End page", min_value=1, value=50, step=1)

theory_ranges = parse_ranges(theory_txt)
practical_ranges = parse_ranges(practical_txt)

st.subheader("1) ملف الطلاب (Roster)")
roster_file = st.file_uploader("Excel/CSV: student_code, student_name", type=["xlsx","xls","csv"])
roster = {}
if roster_file:
    name = roster_file.name.lower()
    df = pd.read_csv(roster_file) if name.endswith(".csv") else pd.read_excel(roster_file)
    df.columns = [c.strip().lower() for c in df.columns]

    if "student_code" not in df.columns or "student_name" not in df.columns:
        st.error("ملف roster لازم يحتوي عمودين: student_code و student_name")
        st.stop()

    df["student_code"] = df["student_code"].astype(str).str.strip()
    df["student_name"] = df["student_name"].astype(str).str.strip()
    roster = dict(zip(df["student_code"], df["student_name"]))
    st.success(f"تم تحميل {len(roster)} طالب")

st.subheader("2) Answer Key")
key_file = st.file_uploader("PDF صفحة واحدة أو صورة", type=["pdf","png","jpg","jpeg"])

st.subheader("3) أوراق الطلاب")
sheets_file = st.file_uploader("PDF متعدد الصفحات أو صورة", type=["pdf","png","jpg","jpeg"])

def load_key_page(file) -> Image.Image:
    b = file.getvalue()
    n = file.name.lower()
    if n.endswith(".pdf"):
        pages = convert_from_bytes(b, dpi=dpi, first_page=1, last_page=1)
        return pages[0]
    return load_single_image(b)

if st.button("🚀 ابدأ التصحيح (Batch)"):
    if not (roster_file and key_file and sheets_file):
        st.error("ارفع جميع الملفات")
        st.stop()

    # مفتاح الإجابة
    key_img = load_key_page(key_file)
    key_thr = preprocess(pil_to_cv(key_img))
    key_ans = read_answers(key_thr, CFG, choices)

    results = []
    prog = st.progress(0)
    total = int(end_page - start_page + 1)

    sf_name = sheets_file.name.lower()
    sf_bytes = sheets_file.getvalue()

    if sf_name.endswith(".pdf"):
        pages_iter = iter_pdf_pages(sf_bytes, dpi=dpi, start=int(start_page), end=int(end_page))
    else:
        # صورة واحدة
        pages_iter = [load_single_image(sf_bytes)]

    for idx, pg in enumerate(pages_iter, 1):
        thr = preprocess(pil_to_cv(pg))
        code = read_student_code(thr, CFG)
        stu_name = roster.get(code, "")

        stu = read_answers(thr, CFG, choices)
        score = 0
        for q, (ka, _) in key_ans.items():
            sa, _ = stu.get(q, ("?",""))
            if theory_ranges and in_ranges(q, theory_ranges) and sa == ka:
                score += 1
            if practical_ranges and in_ranges(q, practical_ranges) and sa == ka:
                score += 1

        results.append({
            "sheet_index": int(start_page) + idx - 1,
            "student_code": code,
            "student_name": stu_name,
            "score": score
        })

        prog.progress(int(idx/total*100))

    out = pd.DataFrame(results)
    buf = io.BytesIO()
    out.to_excel(buf, index=False)
    st.success("تم إنشاء Excel")
    st.download_button("تحميل Excel", buf.getvalue(), f"results_{start_page}_{end_page}.xlsx")
