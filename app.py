import io
import re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw

from streamlit_image_coordinates import streamlit_image_coordinates


# =========================
# Data structures
# =========================
@dataclass
class QBlock:
    x: int
    y: int
    w: int
    h: int
    start_q: int
    end_q: int
    rows: int

@dataclass
class TemplateConfig:
    id_roi: Tuple[int, int, int, int]          # (x,y,w,h)
    id_digits: int
    id_rows: int                               # usually 10 (0..9)
    q_blocks: List[QBlock]

def default_cfg():
    return TemplateConfig(
        id_roi=(0, 0, 0, 0),
        id_digits=4,
        id_rows=10,
        q_blocks=[]
    )

# =========================
# Image utils
# =========================
def load_pages(file_bytes: bytes, name: str) -> List[Image.Image]:
    if name.lower().endswith(".pdf"):
        return convert_from_bytes(file_bytes)
    return [Image.open(io.BytesIO(file_bytes))]

def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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

def pick_one(scores, min_fill, min_ratio):
    # scores: [(choice, score), ...]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0

    if top_s < min_fill:
        return "?", "BLANK"

    if second_s > 0 and (top_s / (second_s + 1e-6)) < min_ratio:
        return "!", "DOUBLE"

    return top_c, "OK"

def parse_ranges(txt: str) -> List[Tuple[int, int]]:
    if not txt.strip():
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

def draw_rect_preview(img: Image.Image, rects: List[Tuple[int,int,int,int]], labels: List[str]) -> Image.Image:
    out = img.copy().convert("RGBA")
    dr = ImageDraw.Draw(out)
    for (x,y,w,h), lab in zip(rects, labels):
        dr.rectangle([x, y, x+w, y+h], outline=(255,0,0,255), width=4)
        dr.text((x+5, y+5), lab, fill=(255,0,0,255))
    return out

# =========================
# Reading ID + answers using cfg
# =========================
def read_student_code(thr: np.ndarray, cfg: TemplateConfig, min_fill=250, min_ratio=1.25) -> str:
    x, y, w, h = cfg.id_roi
    if w <= 0 or h <= 0:
        return ""
    roi = thr[y:y+h, x:x+w]
    rows, cols = cfg.id_rows, cfg.id_digits
    ch, cw = h // rows, w // cols

    digits = []
    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            scores.append((str(r), score_cell(cell)))
        d, stt = pick_one(scores, min_fill, min_ratio)
        digits.append("" if d in ["?","!"] else d)
    return "".join(digits)

def read_answers(thr: np.ndarray, cfg: TemplateConfig, choices: int, min_fill=180, min_ratio=1.25) -> Dict[int, Tuple[str,str]]:
    letters = "ABCDE"[:choices]
    out = {}

    for b in cfg.q_blocks:
        x, y, w, h = b.x, b.y, b.w, b.h
        roi = thr[y:y+h, x:x+w]

        rows = b.rows
        rh = h // rows
        cw = w // choices

        q = b.start_q
        for r in range(rows):
            if q > b.end_q:
                break
            scores = []
            for c in range(choices):
                cell = roi[r*rh:(r+1)*rh, c*cw:(c+1)*cw]
                scores.append((letters[c], score_cell(cell)))
            a, stt = pick_one(scores, min_fill, min_ratio)
            out[q] = (a, stt)
            q += 1

    return out

# =========================
# Streamlit
# =========================
st.set_page_config(page_title="OMR Bubble Sheet (Cloud)", layout="wide")
st.title("✅ OMR Bubble Sheet — اختيار مناطق مثل Remark (بدون Canvas المعطوب)")

# session state
if "cfg" not in st.session_state:
    st.session_state.cfg = default_cfg()
if "pt1" not in st.session_state:
    st.session_state.pt1 = None
if "template_img" not in st.session_state:
    st.session_state.template_img = None

left, right = st.columns([1.4, 1])

with right:
    st.subheader("⚙️ إعدادات")
    choices = st.radio("عدد الخيارات", [4, 5], horizontal=True)
    st.session_state.cfg.id_digits = st.number_input("عدد خانات كود الطالب", min_value=3, max_value=12, value=st.session_state.cfg.id_digits)
    st.session_state.cfg.id_rows = st.number_input("عدد صفوف أرقام الكود (عادة 10)", min_value=5, max_value=15, value=st.session_state.cfg.id_rows)

    st.markdown("---")
    st.subheader("إضافة بلوك أسئلة")
    start_q = st.number_input("Start Q", min_value=1, value=1)
    end_q = st.number_input("End Q", min_value=1, value=20)
    rows_in_block = st.number_input("Rows داخل البلوك", min_value=5, max_value=200, value=20)

    st.markdown("---")
    draw_mode = st.radio("ماذا تحدد الآن؟", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], index=0)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 مسح آخر نقطة"):
            st.session_state.pt1 = None
    with c2:
        if st.button("♻️ Reset الكل"):
            st.session_state.cfg = default_cfg()
            st.session_state.pt1 = None

    st.markdown("---")
    st.caption("ℹ️ طريقة التحديد: انقر على زاوية 1 ثم زاوية 2 (مستطيل).")

with left:
    st.subheader("1) ارفع Template (ورقة نموذج)")
    template_file = st.file_uploader("PDF صفحة واحدة أو صورة", type=["pdf", "png", "jpg", "jpeg"], key="template")

    if template_file:
        pages = load_pages(template_file.getvalue(), template_file.name)
        # أول صفحة فقط كـ template
        st.session_state.template_img = pages[0].convert("RGB")

    if st.session_state.template_img is None:
        st.info("ارفع Template حتى نبدأ التحديد.")
        st.stop()

    # preview with existing rects
    rects = []
    labels = []
    cfg = st.session_state.cfg

    if cfg.id_roi[2] > 0 and cfg.id_roi[3] > 0:
        rects.append(cfg.id_roi)
        labels.append("ID")

    for i, b in enumerate(cfg.q_blocks, 1):
        rects.append((b.x, b.y, b.w, b.h))
        labels.append(f"Q{i}:{b.start_q}-{b.end_q}")

    preview = draw_rect_preview(st.session_state.template_img, rects, labels) if rects else st.session_state.template_img

    st.markdown("### اضغط على الصورة (نقرتين) لتحديد المستطيل")
    click = streamlit_image_coordinates(preview, key="coords")

    if click is not None:
        x, y = int(click["x"]), int(click["y"])
        if st.session_state.pt1 is None:
            st.session_state.pt1 = (x, y)
        else:
            x1, y1 = st.session_state.pt1
            x2, y2 = x, y
            st.session_state.pt1 = None

            rx, ry = min(x1, x2), min(y1, y2)
            rw, rh = abs(x2 - x1), abs(y2 - y1)

            # ignore tiny rectangles
            if rw < 10 or rh < 10:
                st.warning("المستطيل صغير جدًا، أعد التحديد.")
            else:
                if draw_mode.startswith("ID"):
                    st.session_state.cfg.id_roi = (rx, ry, rw, rh)
                    st.success(f"تم حفظ ID ROI: {(rx, ry, rw, rh)}")
                else:
                    st.session_state.cfg.q_blocks.append(
                        QBlock(x=rx, y=ry, w=rw, h=rh, start_q=int(start_q), end_q=int(end_q), rows=int(rows_in_block))
                    )
                    st.success(f"تم إضافة Q Block: {start_q}-{end_q}")

    st.markdown("---")
    st.subheader("2) ملفات التصحيح")

    roster_file = st.file_uploader("Roster (Excel/CSV): student_code, student_name", type=["xlsx", "xls", "csv"])
    key_file = st.file_uploader("Answer Key (PDF صفحة واحدة أو صورة)", type=["pdf","png","jpg","jpeg"])
    sheets_file = st.file_uploader("أوراق الطلاب (PDF متعدد الصفحات أو صور)", type=["pdf","png","jpg","jpeg"])

    st.markdown("### نطاقات التصحيح (اختياري)")
    theory_txt = st.text_input("نطاق النظري (مثال: 1-70 أو 1-40)", "")
    practical_txt = st.text_input("نطاق العملي (مثال: 1-25)", "")

    theory_ranges = parse_ranges(theory_txt)
    practical_ranges = parse_ranges(practical_txt)

    if st.button("🚀 ابدأ التصحيح"):
        if not (roster_file and key_file and sheets_file):
            st.error("ارفع roster + answer key + أوراق الطلاب")
            st.stop()

        if cfg.id_roi[2] <= 0 or cfg.id_roi[3] <= 0 or len(cfg.q_blocks) == 0:
            st.error("لازم تحدد ID ROI + على الأقل Q Block واحد.")
            st.stop()

        # read roster
        if roster_file.name.lower().endswith(("xlsx","xls")):
            df_r = pd.read_excel(roster_file)
        else:
            df_r = pd.read_csv(roster_file)

        df_r.columns = [c.strip().lower() for c in df_r.columns]
        if "student_code" not in df_r.columns or "student_name" not in df_r.columns:
            st.error("Roster لازم يحتوي عمودين: student_code و student_name")
            st.stop()

        roster = dict(zip(df_r["student_code"].astype(str), df_r["student_name"].astype(str)))

        # read key
        key_pages = load_pages(key_file.getvalue(), key_file.name)
        key_thr = preprocess(pil_to_cv(key_pages[0]))
        key_ans = read_answers(key_thr, cfg, choices)

        # read student sheets
        pages = load_pages(sheets_file.getvalue(), sheets_file.name)
        results = []
        prog = st.progress(0)

        for i, pg in enumerate(pages, 1):
            thr = preprocess(pil_to_cv(pg))
            code = read_student_code(thr, cfg)
            name = roster.get(str(code), "")

            stu = read_answers(thr, cfg, choices)

            score = 0
            # إذا ما كتب نطاقات، صحح كل الأسئلة الموجودة في key_ans
            for q, (ka, _) in key_ans.items():
                sa, stt = stu.get(q, ("?", "MISSING"))
                allowed = True
                if theory_ranges or practical_ranges:
                    allowed = (theory_ranges and in_ranges(q, theory_ranges)) or (practical_ranges and in_ranges(q, practical_ranges))
                if not allowed:
                    continue
                if sa == ka:
                    score += 1

            results.append({
                "sheet_index": i,
                "student_code": code,
                "student_name": name,
                "score": score
            })
            prog.progress(int(i / len(pages) * 100))

        out = pd.DataFrame(results)
        buf = io.BytesIO()
        out.to_excel(buf, index=False)

        st.success("✅ تم إنشاء ملف Excel")
        st.download_button("⬇️ تحميل Excel", buf.getvalue(), "results.xlsx")
