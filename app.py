import io
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image

# ✅ مهم: نسخة FIX تعمل مع Streamlit الجديد (تحل مشكلة image_to_url)
from streamlit_drawable_canvas_fix import st_canvas


# ----------------------------
# Data Models
# ----------------------------
@dataclass
class QBlock:
    x: int
    y: int
    w: int
    h: int
    start_q: int
    end_q: int
    rows: int  # عدد الأسئلة عموديًا داخل البلوك

@dataclass
class TemplateConfig:
    template_w: int
    template_h: int

    # كود الطالب
    id_roi: Tuple[int, int, int, int]  # x,y,w,h
    id_digits: int
    id_rows: int  # عادة 10 (0-9)

    # أسئلة
    q_blocks: List[QBlock]


# ----------------------------
# Helpers
# ----------------------------
def pdf_or_image_to_pages(file_bytes: bytes, filename: str, dpi: int = 200) -> List[Image.Image]:
    name = filename.lower()
    if name.endswith(".pdf"):
        # PDF -> PIL pages
        return convert_from_bytes(file_bytes, dpi=dpi)
    # Image -> single page
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def preprocess_threshold(img_bgr: np.ndarray) -> np.ndarray:
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

def pick_one(scores: List[Tuple[str, int]], min_fill: int, min_ratio: float):
    # scores: [(label, pixels), ...]
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    top_label, top_score = scores_sorted[0]
    second_score = scores_sorted[1][1] if len(scores_sorted) > 1 else 0

    if top_score < min_fill:
        return "?", "BLANK", top_score, second_score

    # Double mark / ambiguous
    if second_score > 0 and (top_score / (second_score + 1e-6)) < min_ratio:
        return "!", "DOUBLE", top_score, second_score

    return top_label, "OK", top_score, second_score

def parse_ranges(txt: str) -> List[Tuple[int, int]]:
    """Accept: '1-40' or '1-40, 45-60' or '7,9,10-12' """
    if not txt or not txt.strip():
        return []
    out = []
    for part in txt.split(","):
        p = part.strip()
        if not p:
            continue
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", p)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.append((min(a, b), max(a, b)))
        elif p.isdigit():
            x = int(p)
            out.append((x, x))
    return out

def in_ranges(q: int, ranges: List[Tuple[int, int]]) -> bool:
    if not ranges:
        return True  # إذا ما حددت نطاق، اعتبر كل الأسئلة
    return any(a <= q <= b for a, b in ranges)

def rect_from_canvas(obj) -> Optional[Tuple[int, int, int, int]]:
    """
    st_canvas json_data objects have: left, top, width, height
    Return integer (x,y,w,h)
    """
    try:
        x = int(obj.get("left", 0))
        y = int(obj.get("top", 0))
        w = int(obj.get("width", 0))
        h = int(obj.get("height", 0))
        if w <= 2 or h <= 2:
            return None
        return (x, y, w, h)
    except Exception:
        return None

def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def scale_roi(roi, src_wh, dst_wh):
    """Scale ROI drawn on template image to another page size."""
    (x, y, w, h) = roi
    sw, sh = src_wh
    dw, dh = dst_wh
    sx = dw / float(sw)
    sy = dh / float(sh)
    return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))

def ensure_session():
    if "cfg" not in st.session_state:
        st.session_state.cfg = None
    if "id_roi" not in st.session_state:
        st.session_state.id_roi = None
    if "q_blocks" not in st.session_state:
        st.session_state.q_blocks = []
    if "template_size" not in st.session_state:
        st.session_state.template_size = None

ensure_session()

# ----------------------------
# OMR Readers (use cfg)
# ----------------------------
def read_student_code(thr: np.ndarray, cfg: TemplateConfig,
                      min_fill: int = 250, min_ratio: float = 1.25) -> Tuple[str, Dict]:
    x, y, w, h = cfg.id_roi
    H, W = thr.shape[:2]
    x, y, w, h = clamp_roi(x, y, w, h, W, H)
    roi = thr[y:y + h, x:x + w]

    rows = cfg.id_rows
    cols = cfg.id_digits
    ch = max(1, h // rows)
    cw = max(1, w // cols)

    digits = []
    dbg = {"per_digit": []}

    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            scores.append((str(r), score_cell(cell)))

        d, status, top, second = pick_one(scores, min_fill, min_ratio)
        dbg["per_digit"].append({"col": c, "pick": d, "status": status, "top": top, "second": second, "scores": scores})

        if d in ["?", "!"]:
            digits.append("")  # نخليها فارغة حتى تعرف أن الكود غير موثوق
        else:
            digits.append(d)

    code = "".join(digits)
    return code, dbg

def read_answers(thr: np.ndarray, block: QBlock, choices: int,
                 min_fill: int = 180, min_ratio: float = 1.25) -> Dict[int, Tuple[str, str]]:
    letters = "ABCDE"[:choices]
    out = {}

    H, W = thr.shape[:2]
    x, y, w, h = clamp_roi(block.x, block.y, block.w, block.h, W, H)
    roi = thr[y:y + h, x:x + w]

    rows = max(1, block.rows)
    rh = max(1, h // rows)
    cw = max(1, w // choices)

    q = block.start_q
    for r in range(rows):
        if q > block.end_q:
            break

        scores = []
        for c in range(choices):
            cell = roi[r * rh:(r + 1) * rh, c * cw:(c + 1) * cw]
            scores.append((letters[c], score_cell(cell)))

        a, status, _, _ = pick_one(scores, min_fill, min_ratio)
        out[q] = (a, status)
        q += 1

    return out

def merge_blocks_answers(thr: np.ndarray, cfg: TemplateConfig, choices: int) -> Dict[int, Tuple[str, str]]:
    all_ans = {}
    for b in cfg.q_blocks:
        all_ans.update(read_answers(thr, b, choices))
    return all_ans


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="OMR BubbleSheet (Remark-style)", layout="wide")
st.title("✅ تصحيح ببل شيت (Remark-Style) — تحديد مناطق الكود والأسئلة بالماوس")

with st.expander("0) إعدادات عامة", expanded=True):
    dpi = st.slider("DPI لتحويل PDF إلى صور (أعلى = أدق لكن أثقل)", 120, 260, 200, 10)
    choices = st.radio("عدد الخيارات لكل سؤال", [4, 5], horizontal=True)
    strict_mode = st.checkbox("وضع صارم: BLANK/DOUBLE تعتبر خطأ", value=True)
    min_fill_id = st.slider("حساسية تظليل كود الطالب (min_fill)", 50, 900, 250, 10)
    min_fill_q = st.slider("حساسية تظليل الإجابات (min_fill)", 30, 600, 180, 10)
    min_ratio = st.slider("تمييز الإجابة الصحيحة عن الثانية (min_ratio)", 1.05, 2.50, 1.25, 0.05)

st.divider()

# 1) Template
st.subheader("1) ارفع نموذج ورقة واحدة (Template) ثم ارسم مناطق القراءة")
template_file = st.file_uploader("Template PDF (صفحة واحدة) أو صورة", type=["pdf", "png", "jpg", "jpeg"], key="template_upl")

template_img = None
if template_file:
    t_pages = pdf_or_image_to_pages(template_file.getvalue(), template_file.name, dpi=dpi)
    template_img = t_pages[0].convert("RGB")
    Wt, Ht = template_img.size
    st.session_state.template_size = (Wt, Ht)

    colA, colB = st.columns([2, 1], gap="large")

    with colB:
        st.markdown("### أدوات الرسم")
        draw_mode = st.radio("ماذا سترسم الآن؟", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], index=0)

        id_digits = st.number_input("عدد خانات كود الطالب", min_value=2, max_value=20, value=4, step=1)
        id_rows = st.number_input("عدد صفوف أرقام الكود (عادة 10)", min_value=5, max_value=15, value=10, step=1)

        st.markdown("### إعداد بلوك الأسئلة (عند اختيار Q Block)")
        start_q = st.number_input("Start Q", 1, 500, 1)
        end_q = st.number_input("End Q", 1, 500, 20)
        rows_in_block = st.number_input("Rows (عدد الأسئلة عموديًا داخل هذا البلوك)", 1, 200, 20)

        if st.button("🧹 مسح كل الإعدادات (ID + Blocks)"):
            st.session_state.id_roi = None
            st.session_state.q_blocks = []
            st.session_state.cfg = None
            st.success("تم المسح")

        st.markdown("---")
        st.markdown("### ما تم حفظه")
        st.write("ID ROI:", st.session_state.id_roi)
        st.write("Q Blocks:", len(st.session_state.q_blocks))
        if st.session_state.q_blocks:
            st.json([asdict(b) for b in st.session_state.q_blocks])

    with colA:
        st.markdown(f"Template image size: **{Wt} x {Ht}**")
        # Canvas
        canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.12)",
            stroke_width=2,
            stroke_color="red",
            background_image=template_img,
            update_streamlit=True,
            height=Ht,
            width=Wt,
            drawing_mode="rect",
            key="canvas_template",
        )

        # Save last rectangle according to mode
        if canvas.json_data is not None and "objects" in canvas.json_data:
            objs = canvas.json_data["objects"]
            if len(objs) > 0:
                last = objs[-1]
                r = rect_from_canvas(last)
                if r:
                    x, y, w, h = clamp_roi(*r, Wt, Ht)

                    if draw_mode.startswith("ID"):
                        st.session_state.id_roi = (x, y, w, h)
                        st.success(f"✅ تم حفظ ID ROI = {st.session_state.id_roi}")

                    else:
                        qb = QBlock(x=x, y=y, w=w, h=h, start_q=int(start_q), end_q=int(end_q), rows=int(rows_in_block))
                        st.session_state.q_blocks.append(qb)
                        st.success(f"✅ تم إضافة Q Block: {qb.start_q}-{qb.end_q}")

    # Build cfg
    if st.session_state.id_roi and len(st.session_state.q_blocks) > 0:
        st.session_state.cfg = TemplateConfig(
            template_w=Wt,
            template_h=Ht,
            id_roi=st.session_state.id_roi,
            id_digits=int(id_digits),
            id_rows=int(id_rows),
            q_blocks=st.session_state.q_blocks
        )

st.divider()

# 2) Roster
st.subheader("2) ملف أسماء الطلاب (Roster)")
st.caption("ارفع Excel/CSV يحتوي عمودين بالضبط: student_code, student_name (أي أحرف كبيرة/صغيرة لا تهم).")
roster_file = st.file_uploader("Roster file", type=["xlsx", "xls", "csv"], key="roster_upl")

roster_map: Dict[str, str] = {}
if roster_file:
    if roster_file.name.lower().endswith((".xlsx", ".xls")):
        df_r = pd.read_excel(roster_file)
    else:
        df_r = pd.read_csv(roster_file)
    df_r.columns = [c.strip().lower() for c in df_r.columns]
    if "student_code" not in df_r.columns or "student_name" not in df_r.columns:
        st.error("لازم الأعمدة تكون: student_code و student_name")
    else:
        roster_map = dict(zip(df_r["student_code"].astype(str).str.strip(), df_r["student_name"].astype(str).str.strip()))
        st.success(f"تم تحميل {len(roster_map)} طالب")

st.divider()

# 3) Answer Key + Student Sheets
st.subheader("3) Answer Key + أوراق الطلاب")
key_file = st.file_uploader("Answer Key (PDF صفحة واحدة أو صورة)", type=["pdf", "png", "jpg", "jpeg"], key="key_upl")
sheets_file = st.file_uploader("Student Sheets (PDF متعدد الصفحات أو صور)", type=["pdf", "png", "jpg", "jpeg"], key="sheets_upl")

theory_txt = st.text_input("نطاق التصحيح (مثال: 1-40 أو 1-70, 101-125) — إذا تركته فارغ: يصحح كل شيء")
ranges = parse_ranges(theory_txt)

review_ambiguous = st.checkbox("تجميع الأسئلة المشكلة (BLANK/DOUBLE) في تقرير", value=True)

st.divider()

# Run grading
if st.button("🚀 ابدأ التصحيح"):
    if st.session_state.cfg is None:
        st.error("لازم أولاً ترسم ID ROI + على الأقل Q Block واحد.")
        st.stop()
    if not roster_file:
        st.error("ارفع ملف Roster.")
        st.stop()
    if not (key_file and sheets_file):
        st.error("ارفع Answer Key وأوراق الطلاب.")
        st.stop()

    cfg = st.session_state.cfg

    # Load key page
    key_pages = pdf_or_image_to_pages(key_file.getvalue(), key_file.name, dpi=dpi)
    key_img = key_pages[0].convert("RGB")
    Wk, Hk = key_img.size

    # Scale cfg from template -> key size (لو اختلفت)
    cfg_key = TemplateConfig(
        template_w=Wk, template_h=Hk,
        id_roi=scale_roi(cfg.id_roi, (cfg.template_w, cfg.template_h), (Wk, Hk)),
        id_digits=cfg.id_digits,
        id_rows=cfg.id_rows,
        q_blocks=[
            QBlock(*scale_roi((b.x, b.y, b.w, b.h), (cfg.template_w, cfg.template_h), (Wk, Hk)),
                   start_q=b.start_q, end_q=b.end_q, rows=b.rows)
            for b in cfg.q_blocks
        ]
    )

    key_thr = preprocess_threshold(pil_to_bgr(key_img))
    key_ans = merge_blocks_answers(key_thr, cfg_key, choices)

    # Load student pages
    pages = pdf_or_image_to_pages(sheets_file.getvalue(), sheets_file.name, dpi=dpi)
    total_pages = len(pages)
    prog = st.progress(0)

    results = []
    issues = []  # ambiguous report

    for idx, pg in enumerate(pages, start=1):
        img = pg.convert("RGB")
        Ws, Hs = img.size

        # Scale cfg from template -> student page size
        cfg_s = TemplateConfig(
            template_w=Ws, template_h=Hs,
            id_roi=scale_roi(cfg.id_roi, (cfg.template_w, cfg.template_h), (Ws, Hs)),
            id_digits=cfg.id_digits,
            id_rows=cfg.id_rows,
            q_blocks=[
                QBlock(*scale_roi((b.x, b.y, b.w, b.h), (cfg.template_w, cfg.template_h), (Ws, Hs)),
                       start_q=b.start_q, end_q=b.end_q, rows=b.rows)
                for b in cfg.q_blocks
            ]
        )

        thr = preprocess_threshold(pil_to_bgr(img))

        code, code_dbg = read_student_code(thr, cfg_s, min_fill=min_fill_id, min_ratio=min_ratio)
        code = str(code).strip()
        student_name = roster_map.get(code, "")

        stu_ans = merge_blocks_answers(thr, cfg_s, choices)

        # Score
        score = 0
        for q, (ka, kst) in key_ans.items():
            if not in_ranges(q, ranges):
                continue
            sa, sst = stu_ans.get(q, ("?", "MISSING"))

            # strict mode: blank/double is wrong
            if strict_mode:
                if sa == ka and sst == "OK":
                    score += 1
            else:
                # non-strict: if detected matches key even if ambiguous count it
                if sa == ka:
                    score += 1

            if review_ambiguous and (sst in ["BLANK", "DOUBLE", "MISSING"]):
                issues.append({
                    "sheet_index": idx,
                    "student_code": code,
                    "student_name": student_name,
                    "question": q,
                    "student_mark": sa,
                    "status": sst,
                    "key": ka
                })

        results.append({
            "sheet_index": idx,
            "student_code": code,
            "student_name": student_name,
            "score": int(score)
        })

        prog.progress(int(idx / total_pages * 100))

    df_out = pd.DataFrame(results)

    # Download Excel
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="results")
        if review_ambiguous:
            pd.DataFrame(issues).to_excel(writer, index=False, sheet_name="issues")

    st.success("✅ تم إنشاء ملف Excel (results + issues).")
    st.download_button("⬇️ تحميل Excel", data=buf.getvalue(), file_name="results.xlsx")

    st.markdown("### معاينة النتائج")
    st.dataframe(df_out, use_container_width=True)

    if review_ambiguous and issues:
        st.markdown("### معاينة المشاكل (BLANK/DOUBLE/MISSING)")
        st.dataframe(pd.DataFrame(issues).head(200), use_container_width=True)
