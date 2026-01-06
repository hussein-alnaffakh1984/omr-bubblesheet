import io
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image

# =========================
# Template Configuration
# =========================
@dataclass
class TemplateConfig:
    # منطقة كود الطالب (x, y, w, h) — عدّلها مرة واحدة حسب نموذجكم
    id_roi: Tuple[int,int,int,int] = (1200, 150, 600, 650)
    id_digits: int = 6           # عدد خانات كود الطالب
    id_rows: int = 10            # الأرقام 0–9

    # مناطق الأسئلة (ثلاثة أعمدة)
    # (x, y, w, h, start_q, end_q)
    q_blocks: List[Tuple[int,int,int,int,int,int]] = None
    block_rows: int = 20         # عدد الأسئلة عموديًا في كل بلوك

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
# Utilities
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
# Read Student Code
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
        d, st = pick_one(scores, min_fill, min_ratio)
        digits.append("" if d in ["?","!"] else d)
    return "".join(digits)

# =========================
# Read Answers
# =========================
def read_answers(thr, cfg, choices, min_fill=180, min_ratio=1.25):
    letters = "ABCDE"[:choices]
    out = {}
    for (x,y,w,h,qs,qe) in cfg.q_blocks:
        roi = thr[y:y+h, x:x+w]
        rows = cfg.block_rows
        rh, cw = h//rows, w//choices
        q = qs
        for r in range(rows):
            if q > qe: break
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
st.title("تصحيح ببل شيت – Cloud (Excel فقط)")

with st.expander("⚙️ إعدادات الامتحان", expanded=True):
    choices = st.radio("عدد الخيارات", [4,5], horizontal=True)
    theory_txt = st.text_input("نطاق النظري (مثال: 1-40)")
    practical_txt = st.text_input("نطاق العملي (اختياري)")
    strict = st.checkbox("وضع صارم (BLANK/DOUBLE = خطأ)", True)

theory_ranges = parse_ranges(theory_txt)
practical_ranges = parse_ranges(practical_txt)

st.subheader("1) ملف الطلاب (Roster)")
roster_file = st.file_uploader("Excel: student_code, student_name", type=["xlsx","xls","csv"])
roster = {}
if roster_file:
    df = pd.read_excel(roster_file) if roster_file.name.endswith(("xlsx","xls")) else pd.read_csv(roster_file)
    df.columns = [c.strip().lower() for c in df.columns]
    roster = dict(zip(df["student_code"].astype(str), df["student_name"].astype(str)))
    st.success(f"تم تحميل {len(roster)} طالب")

st.subheader("2) Answer Key")
key_file = st.file_uploader("PDF صفحة واحدة أو صورة", type=["pdf","png","jpg","jpeg"])

st.subheader("3) أوراق الطلاب")
sheets_file = st.file_uploader("PDF متعدد الصفحات أو صور", type=["pdf","png","jpg","jpeg"])

def load_pages(b: bytes, name: str):
    if name.lower().endswith(".pdf"):
        return convert_from_bytes(b)
    return [Image.open(io.BytesIO(b))]

if st.button("🚀 ابدأ التصحيح"):
    if not (roster_file and key_file and sheets_file):
        st.error("ارفع جميع الملفات")
        st.stop()

    key_pages = load_pages(key_file.getvalue(), key_file.name)
    key_thr = preprocess(pil_to_cv(key_pages[0]))
    key_ans = read_answers(key_thr, CFG, choices)

    pages = load_pages(sheets_file.getvalue(), sheets_file.name)
    results = []
    prog = st.progress(0)

    for i, pg in enumerate(pages, 1):
        thr = preprocess(pil_to_cv(pg))
        code = read_student_code(thr, CFG)
        name = roster.get(code, "")
        stu = read_answers(thr, CFG, choices)

        score = 0
        for q,(ka,_) in key_ans.items():
            sa,_ = stu.get(q, ("?",""))
            if theory_ranges and in_ranges(q, theory_ranges):
                if sa == ka: score += 1
            if practical_ranges and in_ranges(q, practical_ranges):
                if sa == ka: score += 1

        results.append({
            "sheet_index": i,
            "student_code": code,
            "student_name": name,
            "score": score
        })
        prog.progress(int(i/len(pages)*100))

    out = pd.DataFrame(results)
    buf = io.BytesIO()
    out.to_excel(buf, index=False)
    st.success("تم إنشاء Excel")
    st.download_button("تحميل Excel", buf.getvalue(), "results.xlsx")
