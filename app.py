import io
import re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image

from streamlit_drawable_canvas import st_canvas


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
    id_roi: Tuple[int, int, int, int]
    id_digits: int
    id_rows: int
    q_blocks: List[QBlock]


# ----------------------------
# Helpers
# ----------------------------
def pdf_or_image_to_pages(file_bytes: bytes, filename: str, dpi: int = 200) -> List[Image.Image]:
    name = filename.lower()
    if name.endswith(".pdf"):
        return convert_from_bytes(file_bytes, dpi=dpi)
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
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    top_label, top_score = scores_sorted[0]
    second_score = scores_sorted[1][1] if len(scores_sorted) > 1 else 0

    if top_score < min_fill:
        return "?", "BLANK"
    if second_score > 0 and (top_score / (second_score + 1e-6)) < min_ratio:
        return "!", "DOUBLE"
    return top_label, "OK"


def parse_ranges(txt: str) -> List[Tuple[int, int]]:
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
        return True
    return any(a <= q <= b for a, b in ranges)


def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def rect_from_canvas_obj(obj) -> Optional[Tuple[int, int, int, int]]:
    """
    Object shape from streamlit-drawable-canvas:
    left, top, width, height
    """
    try:
        x = float(obj.get("left", 0))
        y = float(obj.get("top", 0))
        w = float(obj.get("width", 0))
        h = float(obj.get("height", 0))
        if w < 3 or h < 3:
            return None
        return int(x), int(y), int(w), int(h)
    except Exception:
        return None


def map_canvas_rect_to_image(rect_canvas, canvas_wh, image_wh):
    x, y, w, h = rect_canvas
    cw, ch = canvas_wh
    iw, ih = image_wh
    sx = iw / float(cw)
    sy = ih / float(ch)
    return int(x * sx), int(y * sy), int(w * sx), int(h * sy)


def scale_roi(roi, src_wh, dst_wh):
    (x, y, w, h) = roi
    sw, sh = src_wh
    dw, dh = dst_wh
    sx = dw / float(sw)
    sy = dh / float(sh)
    return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))


# ----------------------------
# OMR Readers
# ----------------------------
def read_student_code(thr: np.ndarray, cfg: TemplateConfig, min_fill=250, min_ratio=1.25) -> str:
    x, y, w, h = cfg.id_roi
    H, W = thr.shape[:2]
    x, y, w, h = clamp_roi(x, y, w, h, W, H)
    roi = thr[y:y+h, x:x+w]

    rows = cfg.id_rows
    cols = cfg.id_digits
    ch = max(1, h // rows)
    cw = max(1, w // cols)

    digits = []
    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            scores.append((str(r), score_cell(cell)))

        d, stt = pick_one(scores, min_fill, min_ratio)
        digits.append("" if d in ["?", "!"] else d)

    return "".join(digits).strip()


def read_answers_block(thr: np.ndarray, block: QBlock, choices: int, min_fill=180, min_ratio=1.25):
    letters = "ABCDE"[:choices]
    out = {}

    H, W = thr.shape[:2]
    x, y, w, h = clamp_roi(block.x, block.y, block.w, block.h, W, H)
    roi = thr[y:y+h, x:x+w]

    rows = max(1, block.rows)
    rh = max(1, h // rows)
    cw = max(1, w // choices)

    q = block.start_q
    for r in range(rows):
        if q > block.end_q:
            break
        scores = []
        for c in range(choices):
            cell = roi[r*rh:(r+1)*rh, c*cw:(c+1)*cw]
            scores.append((letters[c], score_cell(cell)))
        a, stt = pick_one(scores, min_fill, min_ratio)
        out[q] = (a, stt)
        q += 1
    return out


def read_all_answers(thr: np.ndarray, cfg: TemplateConfig, choices: int, min_fill_q: int, min_ratio: float):
    all_ans = {}
    for b in cfg.q_blocks:
        all_ans.update(read_answers_block(thr, b, choices, min_fill=min_fill_q, min_ratio=min_ratio))
    return all_ans


# ----------------------------
# Session State
# ----------------------------
if "id_roi" not in st.session_state:
    st.session_state.id_roi = None
if "q_blocks" not in st.session_state:
    st.session_state.q_blocks = []
if "cfg" not in st.session_state:
    st.session_state.cfg = None
if "template_size" not in st.session_state:
    st.session_state.template_size = None
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_v1"


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="OMR BubbleSheet (Remark-style)", layout="wide")
st.title("✅ تصحيح ببل شيت — تحديد المناطق بالماوس (مثل Remark)")

with st.expander("0) إعدادات عامة", expanded=True):
    dpi = st.slider("DPI لتحويل PDF", 120, 260, 200, 10)
    choices = st.radio("عدد الخيارات لكل سؤال", [4, 5], horizontal=True)
    strict_mode = st.checkbox("وضع صارم: BLANK/DOUBLE = خطأ", value=True)

    min_fill_id = st.slider("حساسية كود الطالب (min_fill)", 50, 900, 250, 10)
    min_fill_q = st.slider("حساسية الإجابات (min_fill)", 30, 600, 180, 10)
    min_ratio = st.slider("تمييز الإجابة عن الثانية (min_ratio)", 1.05, 2.50, 1.25, 0.05)

st.divider()

# 1) Template
st.subheader("1) Template: ارفع صفحة واحدة ثم ارسم (ID ROI + Q Blocks)")
template_file = st.file_uploader("Template PDF (صفحة واحدة) أو صورة", type=["pdf", "png", "jpg", "jpeg"])

template_img = None
if template_file:
    pages = pdf_or_image_to_pages(template_file.getvalue(), template_file.name, dpi=dpi)
    template_img = pages[0].convert("RGB")
    Wt, Ht = template_img.size
    st.session_state.template_size = (Wt, Ht)

    left, right = st.columns([2, 1], gap="large")

    with right:
        st.markdown("### وضع الرسم")
        draw_mode = st.radio("اختر ما سترسمه الآن", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], index=0)

        id_digits = st.number_input("عدد خانات كود الطالب", 2, 30, 4, 1)
        id_rows = st.number_input("عدد صفوف الأرقام (عادة 10)", 5, 15, 10, 1)

        st.markdown("### إعداد Q Block")
        start_q = st.number_input("Start Q", 1, 2000, 1)
        end_q = st.number_input("End Q", 1, 2000, 20)
        rows_in_block = st.number_input("Rows داخل البلوك", 1, 400, 20)

        st.markdown("---")

        colx, coly = st.columns(2)
        with colx:
            if st.button("🧹 مسح ID + Blocks"):
                st.session_state.id_roi = None
                st.session_state.q_blocks = []
                st.session_state.cfg = None
                st.success("تم المسح")

        with coly:
            if st.button("🧽 مسح رسومات الكانفس فقط"):
                # تغيير key يعيد تهيئة الكانفس بالكامل
                st.session_state.canvas_key = f"canvas_{np.random.randint(1, 10**9)}"
                st.success("تم مسح رسومات الكانفس")

        st.markdown("### ما تم حفظه")
        st.write("ID ROI:", st.session_state.id_roi)
        st.write("Q Blocks:", len(st.session_state.q_blocks))
        if st.session_state.q_blocks:
            st.json([asdict(b) for b in st.session_state.q_blocks])

    with left:
        st.markdown(f"**حجم صورة الـTemplate:** {Wt} × {Ht}")
        st.image(template_img, caption="Template", use_container_width=True)

        st.warning("✍️ الرسم يتم على المربع الأبيض (Canvas) أدناه. اسحب بالماوس لرسم مستطيل.")

        canvas_w = st.slider("Canvas width (لا تغيّره بعد ما ترسم)", 700, 1400, min(1000, Wt), 10)
        canvas_h = int(canvas_w * (Ht / Wt))

        canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.15)",
            stroke_width=3,
            stroke_color="#ff0000",
            background_color="rgba(255,255,255,1)",  # ✅ خلفية بيضاء تجعل الرسم واضح ويشتغل
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="rect",
            key=st.session_state.canvas_key,
        )

        # حفظ آخر مستطيل
        if canvas.json_data and "objects" in canvas.json_data and len(canvas.json_data["objects"]) > 0:
            last = canvas.json_data["objects"][-1]
            rc = rect_from_canvas_obj(last)
            if rc:
                # تحويل من canvas coords -> image coords
                ri = map_canvas_rect_to_image(rc, (canvas_w, canvas_h), (Wt, Ht))
                x, y, w, h = clamp_roi(*ri, Wt, Ht)

                if draw_mode.startswith("ID"):
                    st.session_state.id_roi = (x, y, w, h)
                    st.success(f"✅ حفظ ID ROI = {st.session_state.id_roi}")
                else:
                    qb = QBlock(x=x, y=y, w=w, h=h, start_q=int(start_q), end_q=int(end_q), rows=int(rows_in_block))
                    st.session_state.q_blocks.append(qb)
                    st.success(f"✅ إضافة Q Block: {qb.start_q}-{qb.end_q}")

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
st.subheader("2) Roster (student_code, student_name)")
roster_file = st.file_uploader("Excel/CSV", type=["xlsx", "xls", "csv"], key="roster")

roster_map: Dict[str, str] = {}
if roster_file:
    if roster_file.name.lower().endswith((".xlsx", ".xls")):
        df_r = pd.read_excel(roster_file)
    else:
        df_r = pd.read_csv(roster_file)

    df_r.columns = [c.strip().lower() for c in df_r.columns]
    if "student_code" not in df_r.columns or "student_name" not in df_r.columns:
        st.error("لازم الأعمدة: student_code و student_name")
    else:
        roster_map = dict(zip(df_r["student_code"].astype(str).str.strip(),
                              df_r["student_name"].astype(str).str.strip()))
        st.success(f"تم تحميل {len(roster_map)} طالب")

st.divider()

# 3) Key + Sheets
st.subheader("3) Answer Key + Student Sheets")
key_file = st.file_uploader("Answer Key (PDF صفحة واحدة أو صورة)", type=["pdf", "png", "jpg", "jpeg"], key="key")
sheets_file = st.file_uploader("Student Sheets (PDF متعدد الصفحات أو صور)", type=["pdf", "png", "jpg", "jpeg"], key="sheets")

ranges_txt = st.text_input("نطاق التصحيح (اختياري) مثال: 1-40 أو 1-70,101-125 (فارغ = كل الأسئلة)")
ranges = parse_ranges(ranges_txt)

review_issues = st.checkbox("إخراج تقرير issues (BLANK/DOUBLE/MISSING)", value=True)

st.divider()

if st.button("🚀 ابدأ التصحيح"):
    if st.session_state.cfg is None:
        st.error("ارسم ID ROI + على الأقل Q Block واحد.")
        st.stop()
    if not roster_map:
        st.error("ارفع Roster صحيح.")
        st.stop()
    if not (key_file and sheets_file):
        st.error("ارفع Answer Key وأوراق الطلاب.")
        st.stop()

    cfg = st.session_state.cfg

    # Key
    key_pages = pdf_or_image_to_pages(key_file.getvalue(), key_file.name, dpi=dpi)
    key_img = key_pages[0].convert("RGB")
    Wk, Hk = key_img.size

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
    key_ans = read_all_answers(key_thr, cfg_key, choices, min_fill_q=min_fill_q, min_ratio=min_ratio)

    # Student pages
    pages = pdf_or_image_to_pages(sheets_file.getvalue(), sheets_file.name, dpi=dpi)
    total = len(pages)

    results = []
    issues = []
    prog = st.progress(0)

    for idx, pg in enumerate(pages, start=1):
        img = pg.convert("RGB")
        Ws, Hs = img.size

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

        code = read_student_code(thr, cfg_s, min_fill=min_fill_id, min_ratio=min_ratio)
        name = roster_map.get(code, "")

        stu_ans = read_all_answers(thr, cfg_s, choices, min_fill_q=min_fill_q, min_ratio=min_ratio)

        score = 0
        for q, (ka, _) in key_ans.items():
            if not in_ranges(q, ranges):
                continue
            sa, stt = stu_ans.get(q, ("?", "MISSING"))

            if strict_mode:
                if sa == ka and stt == "OK":
                    score += 1
            else:
                if sa == ka:
                    score += 1

            if review_issues and stt in ["BLANK", "DOUBLE", "MISSING"]:
                issues.append({
                    "sheet_index": idx,
                    "student_code": code,
                    "student_name": name,
                    "question": q,
                    "student_mark": sa,
                    "status": stt,
                    "key": ka
                })

        results.append({
            "sheet_index": idx,
            "student_code": code,
            "student_name": name,
            "score": int(score)
        })

        prog.progress(int(idx / total * 100))

    df_out = pd.DataFrame(results)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="results")
        if review_issues:
            pd.DataFrame(issues).to_excel(writer, index=False, sheet_name="issues")

    st.success("✅ تم إنشاء ملف Excel.")
    st.download_button("⬇️ تحميل Excel", data=buf.getvalue(), file_name="results.xlsx")

    st.dataframe(df_out, use_container_width=True)
    if review_issues and len(issues) > 0:
        st.markdown("### issues (أول 200 صف)")
        st.dataframe(pd.DataFrame(issues).head(200), use_container_width=True)
