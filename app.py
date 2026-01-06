# app.py  (FULL REPLACEMENT - Robust OMR + Student Code + Excel Export)
# --------------------------------------------------------------------
# ✅ Fill-Ratio based detection (stable across DPI)
# ✅ Student code never becomes empty (uses ? for uncertain digits)
# ✅ Adds id_status column to diagnose problems
# ✅ Calibration sliders for ROI
# ✅ Batch pages for Streamlit Cloud

import io
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
import cv2

# =========================
# Reference sheet size
# =========================
REF_W, REF_H = 991, 1420  # ثابت على نموذجك

@dataclass
class TemplateConfig:
    id_roi: Tuple[int, int, int, int]  # x,y,w,h
    id_digits: int
    id_rows: int
    q_blocks: List[Tuple[int, int, int, int, int, int]]  # x,y,w,h, qs,qe
    block_rows: int

# ✅ إعدادات أساسية مناسبة لنموذجك (يمكن تحريكها بالسلايدر)
BASE = TemplateConfig(
    id_roi=(600, 140, 320, 420),  # منطقة كود الطالب
    id_digits=4,
    id_rows=10,
    q_blocks=[
        (120, 520, 260, 860, 1, 20),
        (440, 520, 260, 860, 21, 40),
        (760, 520, 220, 860, 41, 60),
    ],
    block_rows=20
)

def scale_box(box, sx, sy):
    x, y, w, h = box
    return (int(x*sx), int(y*sy), int(w*sx), int(h*sy))

def cfg_for_image(img_w, img_h, id_offset=(0,0,0,0), b_offset=(0,0,0,0)):
    sx, sy = img_w / REF_W, img_h / REF_H

    ix, iy, iw, ih = BASE.id_roi
    ox, oy, ow, oh = id_offset
    id_ref = (ix + ox, iy + oy, iw + ow, ih + oh)
    id_scaled = scale_box(id_ref, sx, sy)

    bx, by, bw, bh = b_offset
    blocks_scaled = []
    for (x, y, w, h, qs, qe) in BASE.q_blocks:
        ref = (x + bx, y + by, w + bw, h + bh)
        xs, ys, ws, hs = scale_box(ref, sx, sy)
        blocks_scaled.append((xs, ys, ws, hs, qs, qe))

    return TemplateConfig(
        id_roi=id_scaled,
        id_digits=BASE.id_digits,
        id_rows=BASE.id_rows,
        q_blocks=blocks_scaled,
        block_rows=BASE.block_rows
    )

# =========================
# Preprocess
# =========================
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def preprocess(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )
    return thr

def cell_fill_ratio(bin_img: np.ndarray) -> float:
    return float((bin_img > 0).mean())

def pick_one_ratio(scores: List[Tuple[str, float]], blank_thr: float, min_ratio: float):
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0.0

    if top_s < blank_thr:
        return "?", "BLANK"

    if second_s > 0 and (top_s / (second_s + 1e-9)) < min_ratio:
        return "!", "DOUBLE"

    return top_c, "OK"

# =========================
# Readers
# =========================
def read_student_code(thr: np.ndarray, cfg: TemplateConfig,
                      blank_thr: float, min_ratio: float):
    """
    يرجع كود مثل 1234 أو 12?4 إذا عنده مشكلة، وليس فارغ.
    ويرجع statuses لكل خانة.
    """
    x, y, w, h = cfg.id_roi
    roi = thr[y:y+h, x:x+w]

    rows, cols = cfg.id_rows, cfg.id_digits
    ch, cw = h // rows, w // cols

    digits = []
    statuses = []
    for c in range(cols):
        col_scores = []
        for r in range(rows):
            cell = roi[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            col_scores.append((str(r), cell_fill_ratio(cell)))

        d, stt = pick_one_ratio(col_scores, blank_thr=blank_thr, min_ratio=min_ratio)

        # ✅ لا تخليها فارغ: إذا مشكلة خلي '?'
        digits.append(d if d not in ["?","!"] else "?")
        statuses.append(stt)

    return "".join(digits), statuses

def read_answers(thr: np.ndarray, cfg: TemplateConfig, choices: int,
                 blank_thr: float, min_ratio: float):
    letters = "ABCDE"[:choices]
    out: Dict[int, Tuple[str, str]] = {}

    for (x, y, w, h, qs, qe) in cfg.q_blocks:
        roi = thr[y:y+h, x:x+w]
        rows = cfg.block_rows
        rh, cw = h // rows, w // choices

        q = qs
        for r in range(rows):
            if q > qe:
                break

            scores = []
            for c in range(choices):
                cell = roi[r*rh:(r+1)*rh, c*cw:(c+1)*cw]
                scores.append((letters[c], cell_fill_ratio(cell)))

            a, stt = pick_one_ratio(scores, blank_thr=blank_thr, min_ratio=min_ratio)
            out[q] = (a, stt)
            q += 1

    return out

# =========================
# Scoring helpers
# =========================
def parse_ranges(txt: str):
    if not txt or not txt.strip():
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

def in_ranges(q: int, ranges):
    return any(a <= q <= b for a,b in ranges)

def compute_score(key_ans, stu_ans, theory_ranges, practical_ranges, strict=False):
    """
    strict=False: نتجاهل BLANK/DOUBLE (لا تزيد ولا تنقص)
    strict=True : BLANK/DOUBLE تعتبر خطأ أيضًا (نفس النتيجة عمليًا لأنها لن تساوي المفتاح)
    """
    score = 0
    grade_all = (not theory_ranges and not practical_ranges)

    for q, (ka, _) in key_ans.items():
        sa, stt = stu_ans.get(q, ("?","BLANK"))

        # إذا ماكو اختيار واضح
        if sa in ["?","!"]:
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

# =========================
# PDF loaders
# =========================
def load_pages(file_bytes: bytes, filename: str, dpi: int, first_page=1, last_page=1):
    if filename.lower().endswith(".pdf"):
        return convert_from_bytes(file_bytes, dpi=dpi, first_page=first_page, last_page=last_page)
    return [Image.open(io.BytesIO(file_bytes))]

def crop_preview(thr, box):
    x, y, w, h = box
    H, W = thr.shape[:2]
    x = max(0, min(x, W-1))
    y = max(0, min(y, H-1))
    w = max(1, min(w, W-x))
    h = max(1, min(h, H-y))
    return thr[y:y+h, x:x+w]

# =========================
# UI
# =========================
st.set_page_config(page_title="OMR Bubble Sheet", layout="wide")
st.title("تصحيح ببل شيت – (قراءة كود + تصحيح + Excel)")

c1, c2 = st.columns([1,1])

with c1:
    st.subheader("⚙️ إعدادات")
    choices = st.radio("عدد الخيارات", [4,5], horizontal=True)
    dpi = st.slider("DPI تحويل PDF", 80, 200, 120, 10)

    theory_txt = st.text_input("نطاق النظري (مثال: 1-60) — إذا فارغ يصحح الكل", "")
    practical_txt = st.text_input("نطاق العملي (اختياري)", "")

    theory_ranges = parse_ranges(theory_txt)
    practical_ranges = parse_ranges(practical_txt)

    strict = st.checkbox("Strict (BLANK/DOUBLE = لا تُحسب)", False)

    st.markdown("---")
    st.subheader("🎛 حساسية (بالنسبة)")
    # ✅ أهم شيء للكود: خلي min_ratio منخفض (1.05–1.15)
    id_blank_thr = st.slider("ID blank threshold", 0.01, 0.40, 0.06, 0.01)
    id_min_ratio  = st.slider("ID double ratio",    1.05, 2.50, 1.10, 0.05)

    ans_blank_thr = st.slider("Answers blank threshold", 0.01, 0.40, 0.08, 0.01)
    ans_min_ratio = st.slider("Answers double ratio",    1.05, 2.50, 1.20, 0.05)

    st.markdown("---")
    st.subheader("🧩 معايرة ROI")
    debug = st.checkbox("Debug (عرض قصّ الصفحة الأولى)", True)

    id_dx = st.slider("ID x offset", -200, 200, 0, 5)
    id_dy = st.slider("ID y offset", -200, 200, 0, 5)
    id_dw = st.slider("ID w offset", -200, 200, 0, 5)
    id_dh = st.slider("ID h offset", -200, 200, 0, 5)

    b_dx = st.slider("Blocks x offset", -200, 200, 0, 5)
    b_dy = st.slider("Blocks y offset", -200, 200, 0, 5)
    b_dw = st.slider("Blocks w offset", -200, 200, 0, 5)
    b_dh = st.slider("Blocks h offset", -200, 200, 0, 5)

    st.markdown("---")
    st.subheader("📄 Batch")
    start_page = st.number_input("Start page", min_value=1, value=1, step=1)
    end_page   = st.number_input("End page",   min_value=1, value=100, step=1)

with c2:
    st.subheader("📥 الملفات")
    roster_file = st.file_uploader("1) Roster: student_code, student_name", type=["xlsx","xls","csv"])
    key_file    = st.file_uploader("2) Answer Key (PDF صفحة واحدة/صورة)", type=["pdf","png","jpg","jpeg"])
    sheets_file = st.file_uploader("3) Student Sheets (PDF)", type=["pdf","png","jpg","jpeg"])

# Load roster
roster = {}
if roster_file:
    fn = roster_file.name.lower()
    df = pd.read_csv(roster_file) if fn.endswith(".csv") else pd.read_excel(roster_file)
    df.columns = [c.strip().lower() for c in df.columns]

    if "student_code" not in df.columns or "student_name" not in df.columns:
        st.error("Roster لازم يحتوي: student_code و student_name")
        st.stop()

    df["student_code"] = df["student_code"].astype(str).str.strip()
    df["student_name"] = df["student_name"].astype(str).str.strip()

    # ✅ مهم جدًا: أكوادك 4 خانات
    df["student_code"] = df["student_code"].apply(lambda x: x.zfill(BASE.id_digits))

    roster = dict(zip(df["student_code"], df["student_name"]))
    st.success(f"✅ Loaded roster: {len(roster)} students")

if st.button("🚀 ابدأ التصحيح"):
    if not (roster_file and key_file and sheets_file):
        st.error("ارفع Roster + Answer Key + Student Sheets")
        st.stop()
    if end_page < start_page:
        st.error("End page لازم >= Start page")
        st.stop()

    # --- Key ---
    key_pages = load_pages(key_file.getvalue(), key_file.name, dpi=dpi, first_page=1, last_page=1)
    key_thr = preprocess(pil_to_bgr(key_pages[0]))
    Hk, Wk = key_thr.shape[:2]

    cfgk = cfg_for_image(Wk, Hk,
                         id_offset=(id_dx,id_dy,id_dw,id_dh),
                         b_offset=(b_dx,b_dy,b_dw,b_dh))

    key_ans = read_answers(key_thr, cfgk, choices,
                           blank_thr=ans_blank_thr, min_ratio=ans_min_ratio)

    # --- Student pages ---
    if sheets_file.name.lower().endswith(".pdf"):
        pages = load_pages(sheets_file.getvalue(), sheets_file.name, dpi=dpi,
                           first_page=int(start_page), last_page=int(end_page))
    else:
        pages = [Image.open(io.BytesIO(sheets_file.getvalue()))]

    results = []
    prog = st.progress(0)
    total = len(pages)

    for i, pg in enumerate(pages, 1):
        sheet_index = int(start_page) + i - 1

        thr = preprocess(pil_to_bgr(pg))
        H, W = thr.shape[:2]
        cfg = cfg_for_image(W, H,
                            id_offset=(id_dx,id_dy,id_dw,id_dh),
                            b_offset=(b_dx,b_dy,b_dw,b_dh))

        code, code_status = read_student_code(thr, cfg,
                                              blank_thr=id_blank_thr,
                                              min_ratio=id_min_ratio)

        # ✅ لو بيه ? ما نطابق الاسم (لأنه غير مؤكد)
        name = ""
        if "?" not in code:
            name = roster.get(code, "")

        stu_ans = read_answers(thr, cfg, choices,
                               blank_thr=ans_blank_thr, min_ratio=ans_min_ratio)

        score = compute_score(key_ans, stu_ans,
                              theory_ranges, practical_ranges,
                              strict=strict)

        # Debug first student page
        if debug and i == 1:
            st.markdown("### 🔎 Debug (First Student Page)")
            st.image(crop_preview(thr, cfg.id_roi),
                     caption=f"ID ROI | code={code} | status={code_status}", clamp=True)

            for bi, (x,y,w,h,qs,qe) in enumerate(cfg.q_blocks, 1):
                st.image(thr[y:y+h, x:x+w], caption=f"Block {bi} ({qs}-{qe})", clamp=True)

            st.write("Key first 10:", {k: key_ans[k] for k in sorted(key_ans)[:10]})
            st.write("Student first 10:", {k: stu_ans[k] for k in sorted(stu_ans)[:10]})

        results.append({
            "sheet_index": sheet_index,
            "student_code": code,
            "student_name": name,
            "score": score,
            "id_status": ",".join(code_status)
        })

        prog.progress(int(i/total*100))

    out = pd.DataFrame(results)
    buf = io.BytesIO()
    out.to_excel(buf, index=False)

    st.success("✅ تم إنشاء Excel")
    st.download_button(
        "⬇️ تحميل Excel",
        data=buf.getvalue(),
        file_name=f"results_{start_page}_{end_page}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
