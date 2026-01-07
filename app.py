# app.py
import io
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image

from streamlit_drawable_canvas import st_canvas

# ----------------------------
# Helpers
# ----------------------------
def load_pages(file_bytes: bytes, filename: str) -> List[Image.Image]:
    name = filename.lower()
    if name.endswith(".pdf"):
        # poppler-utils required on Streamlit Cloud
        pages = convert_from_bytes(file_bytes, dpi=200)
        return [p.convert("RGB") for p in pages]
    else:
        return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

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

def pick_one(scores: List[Tuple[str, int]], min_fill: int, min_ratio: float) -> Tuple[str, str]:
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0

    if top_s < min_fill:
        return "?", "BLANK"
    if second_s > 0 and (top_s / (second_s + 1e-6)) < min_ratio:
        return "!", "DOUBLE"
    return top_c, "OK"

def parse_ranges(txt: str) -> List[Tuple[int, int]]:
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

def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h

# ----------------------------
# Template data
# ----------------------------
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
    # ID grid inside ROI
    id_roi: Optional[Tuple[int, int, int, int]] = None
    id_digits: int = 4        # you can change
    id_rows: int = 10         # 0..9
    # Question blocks
    q_blocks: List[QBlock] = None

def cfg_init() -> TemplateConfig:
    return TemplateConfig(id_roi=None, id_digits=4, id_rows=10, q_blocks=[])

# ----------------------------
# OMR Readers (grid based)
# ----------------------------
def read_student_code(thr: np.ndarray, cfg: TemplateConfig, min_fill=250, min_ratio=1.25) -> str:
    if cfg.id_roi is None:
        return ""
    x, y, w, h = cfg.id_roi
    H, W = thr.shape[:2]
    x, y, w, h = clamp_roi(x, y, w, h, W, H)
    roi = thr[y:y+h, x:x+w]

    rows = cfg.id_rows
    cols = cfg.id_digits
    ch = h // rows
    cw = w // cols

    digits = []
    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            scores.append((str(r), score_cell(cell)))
        d, status = pick_one(scores, min_fill=min_fill, min_ratio=min_ratio)
        digits.append("" if d in ["?", "!"] else d)

    code = "".join(digits).strip()
    return code

def read_answers_from_blocks(thr: np.ndarray, cfg: TemplateConfig, choices: int,
                            min_fill=180, min_ratio=1.25) -> Dict[int, Tuple[str, str]]:
    letters = "ABCDE"[:choices]
    out: Dict[int, Tuple[str, str]] = {}
    H, W = thr.shape[:2]

    for blk in (cfg.q_blocks or []):
        x, y, w, h = clamp_roi(blk.x, blk.y, blk.w, blk.h, W, H)
        roi = thr[y:y+h, x:x+w]

        rows = max(1, int(blk.rows))
        rh = h // rows
        cw = w // choices

        q = int(blk.start_q)
        endq = int(blk.end_q)

        for r in range(rows):
            if q > endq:
                break
            scores = []
            for c in range(choices):
                cell = roi[r*rh:(r+1)*rh, c*cw:(c+1)*cw]
                scores.append((letters[c], score_cell(cell)))
            a, status = pick_one(scores, min_fill=min_fill, min_ratio=min_ratio)
            out[q] = (a, status)
            q += 1

    return out

# ----------------------------
# UI state
# ----------------------------
st.set_page_config(page_title="OMR BubbleSheet (Remark-style)", layout="wide")
st.title("✅ تصحيح ببل شيت — واجهة مثل Remark (تحديد يدوي + Excel)")

if "cfg" not in st.session_state:
    st.session_state.cfg = cfg_init()

if "last_drawn_rect" not in st.session_state:
    st.session_state.last_drawn_rect = None

# ----------------------------
# Left: Template drawing
# ----------------------------
st.subheader("1) حمّل نموذج ورقة واحدة (Template) وارسم مناطق الكود والأسئلة")

colL, colR = st.columns([1.35, 1])

with colR:
    st.markdown("### إعدادات عامة")
    st.session_state.cfg.id_digits = st.number_input("عدد خانات كود الطالب", min_value=1, max_value=20, value=int(st.session_state.cfg.id_digits), step=1)
    id_rows = st.number_input("عدد صفوف أرقام الكود (عادة 10)", min_value=5, max_value=15, value=int(st.session_state.cfg.id_rows), step=1)
    st.session_state.cfg.id_rows = int(id_rows)

    choices = st.radio("عدد الخيارات (A..)", [4, 5], horizontal=True)

    st.markdown("---")
    st.markdown("### ماذا ترسم الآن؟")
    draw_mode = st.radio("", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], index=0)

    st.markdown("---")
    st.markdown("### إعداد بلوك الأسئلة قبل الحفظ")
    start_q = st.number_input("Start Q", min_value=1, value=1, step=1)
    end_q = st.number_input("End Q", min_value=1, value=20, step=1)
    blk_rows = st.number_input("Rows داخل البلوك", min_value=1, value=20, step=1)

    st.caption("🔹 ارسم مستطيل واحد، بعدها اضغط زر الحفظ المناسب.")

    if st.button("🧹 Reset الكل"):
        st.session_state.cfg = cfg_init()
        st.session_state.last_drawn_rect = None
        st.rerun()

with colL:
    template_file = st.file_uploader("Template (PDF صفحة واحدة أو صورة)", type=["pdf", "png", "jpg", "jpeg"], key="template_upl")

    if template_file is None:
        st.info("ارفع ملف Template أولاً.")
        st.stop()

    template_pages = load_pages(template_file.getvalue(), template_file.name)
    template_img = template_pages[0]

    # Display size controls (full page, less lag)
    st.markdown("#### عرض الصفحة كاملة")
    canvas_w = st.slider("Canvas width (لا تغيّره بعد ما ترسم)", min_value=700, max_value=1400, value=1100, step=10)
    zoom = st.slider("Zoom", min_value=0.5, max_value=2.0, value=1.0, step=0.05)

    # Resize for display
    W0, H0 = template_img.size
    disp_w = int(canvas_w * zoom)
    scale = disp_w / float(W0)
    disp_h = int(H0 * scale)
    disp_img = template_img.resize((disp_w, disp_h), Image.BILINEAR)

    st.caption(f"Template الأصلية: {W0}×{H0} | المعروضة: {disp_w}×{disp_h} | scale={scale:.4f}")

    # Canvas: IMPORTANT -> only supported args (no realtime_update/update_streamlit/initial_drawing)
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.20)",
        stroke_width=2,
        stroke_color="red",
        background_image=disp_img,     # PIL Image only
        drawing_mode="rect",
        width=disp_w,
        height=disp_h,
        key="canvas_main",
    )

    # Extract last rectangle
    rect = None
    if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
        objs = canvas_result.json_data["objects"]
        if len(objs) > 0:
            # Take the last drawn object
            o = objs[-1]
            # Fabric.js fields: left, top, width, height
            if all(k in o for k in ["left", "top", "width", "height"]):
                rect = (float(o["left"]), float(o["top"]), float(o["width"]), float(o["height"]))
                st.session_state.last_drawn_rect = rect

    st.markdown("### 2) حفظ المستطيل المرسوم")
    if st.session_state.last_drawn_rect is None:
        st.warning("ارسم مستطيل (Rectangle) على الصورة أولاً.")
    else:
        lx, ty, rw, rh = st.session_state.last_drawn_rect
        st.write({"left": lx, "top": ty, "w": rw, "h": rh})

        # Convert display coords -> original image coords
        ox = int(lx / scale)
        oy = int(ty / scale)
        ow = int(rw / scale)
        oh = int(rh / scale)

        if draw_mode.startswith("ID"):
            if st.button("💾 احفظ ID ROI"):
                st.session_state.cfg.id_roi = (ox, oy, ow, oh)
                st.success(f"تم حفظ ID ROI: {st.session_state.cfg.id_roi}")

        else:
            if st.button("💾 أضف Q Block"):
                blk = QBlock(x=ox, y=oy, w=ow, h=oh, start_q=int(start_q), end_q=int(end_q), rows=int(blk_rows))
                st.session_state.cfg.q_blocks.append(blk)
                st.success(f"تمت إضافة Q Block: {asdict(blk)}")

    st.markdown("---")
    st.markdown("### الإعدادات الحالية (Template Config)")
    st.json({
        "id_roi": st.session_state.cfg.id_roi,
        "id_digits": st.session_state.cfg.id_digits,
        "id_rows": st.session_state.cfg.id_rows,
        "q_blocks": [asdict(b) for b in (st.session_state.cfg.q_blocks or [])]
    })

# ----------------------------
# Grading section
# ----------------------------
st.subheader("3) ملفات التصحيح")

with st.expander("⚙️ نطاق التصحيح (اختر أي جزء تريد)", expanded=True):
    theory_txt = st.text_input("نطاق النظري (مثال: 1-40 أو 1-70)", value="")
    practical_txt = st.text_input("نطاق العملي (اختياري مثال: 1-25)", value="")
    strict = st.checkbox("Strict: BLANK/DOUBLE تُحسب خطأ", True)

theory_ranges = parse_ranges(theory_txt)
practical_ranges = parse_ranges(practical_txt)

st.markdown("#### (A) ملف الأسماء (Roster)")
roster_file = st.file_uploader("Excel/CSV يحتوي: student_code, student_name", type=["xlsx", "xls", "csv"], key="roster_upl")

roster: Dict[str, str] = {}
if roster_file is not None:
    if roster_file.name.lower().endswith(("xlsx", "xls")):
        df = pd.read_excel(roster_file)
    else:
        df = pd.read_csv(roster_file)

    df.columns = [c.strip().lower() for c in df.columns]
    if "student_code" in df.columns and "student_name" in df.columns:
        roster = dict(zip(df["student_code"].astype(str).str.strip(), df["student_name"].astype(str)))
        st.success(f"تم تحميل roster: {len(roster)} طالب")
    else:
        st.error("يجب أن يحتوي الملف على عمودين: student_code و student_name")

st.markdown("#### (B) Answer Key")
key_file = st.file_uploader("Answer Key (PDF صفحة واحدة أو صورة)", type=["pdf", "png", "jpg", "jpeg"], key="key_upl")

st.markdown("#### (C) أوراق الطلاب")
sheets_file = st.file_uploader("Student Sheets (PDF متعدد الصفحات أو صور)", type=["pdf", "png", "jpg", "jpeg"], key="sheets_upl")

def score_one_student(key_ans: Dict[int, Tuple[str, str]],
                      stu_ans: Dict[int, Tuple[str, str]],
                      ranges1: List[Tuple[int, int]],
                      ranges2: List[Tuple[int, int]],
                      strict_mode: bool) -> int:
    score = 0
    # Union scoring by requested ranges (theory + practical)
    for q, (ka, _) in key_ans.items():
        sa, stt = stu_ans.get(q, ("?", "MISSING"))
        in_theory = (len(ranges1) == 0) or in_ranges(q, ranges1)
        in_prac = (len(ranges2) > 0) and in_ranges(q, ranges2)

        # If user provided theory ranges, only score those; practical scored separately if provided
        should_score = False
        if len(ranges1) > 0:
            should_score = in_theory or in_prac
        else:
            # if no theory range entered, score everything + practical if given
            should_score = True

        if not should_score:
            continue

        if strict_mode and stt in ["BLANK", "DOUBLE"]:
            continue
        if sa == ka:
            score += 1
    return score

st.markdown("---")
if st.button("🚀 ابدأ التصحيح الآن", type="primary"):
    if st.session_state.cfg.id_roi is None or not st.session_state.cfg.q_blocks:
        st.error("لازم ترسم وتحفظ ID ROI + على الأقل Q Block واحد.")
        st.stop()
    if key_file is None or sheets_file is None:
        st.error("ارفع Answer Key وأوراق الطلاب.")
        st.stop()

    # Load key
    key_pages = load_pages(key_file.getvalue(), key_file.name)
    key_thr = preprocess(pil_to_cv(key_pages[0]))
    key_ans = read_answers_from_blocks(key_thr, st.session_state.cfg, choices=choices)

    if len(key_ans) == 0:
        st.error("لم أستطع استخراج إجابات من Answer Key. راجع Q Blocks.")
        st.stop()

    # Load student sheets
    pages = load_pages(sheets_file.getvalue(), sheets_file.name)
    prog = st.progress(0)
    results = []

    for i, pg in enumerate(pages, start=1):
        thr = preprocess(pil_to_cv(pg))

        code = read_student_code(thr, st.session_state.cfg)
        name = roster.get(code, "") if roster else ""

        stu_ans = read_answers_from_blocks(thr, st.session_state.cfg, choices=choices)
        score = score_one_student(key_ans, stu_ans, theory_ranges, practical_ranges, strict)

        results.append({
            "sheet_index": i,
            "student_code": code,
            "student_name": name,
            "score": int(score),
        })
        prog.progress(int(i / max(1, len(pages)) * 100))

    out = pd.DataFrame(results)
    st.success("✅ تم التصحيح")
    st.dataframe(out, use_container_width=True)

    buf = io.BytesIO()
    out.to_excel(buf, index=False)
    st.download_button("⬇️ تحميل النتائج Excel", data=buf.getvalue(), file_name="results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
