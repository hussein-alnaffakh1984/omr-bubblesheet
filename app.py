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

from streamlit_drawable_canvas import st_canvas


# =========================
# Helpers
# =========================
def load_pages(file_bytes: bytes, filename: str) -> List[Image.Image]:
    name = filename.lower()
    if name.endswith(".pdf"):
        # dpi منخفض نسبيًا لتسريع السحابة (ممكن ترفعه لاحقًا إذا احتجت دقة أعلى)
        return convert_from_bytes(file_bytes, dpi=200)
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

def parse_ranges(txt: str) -> List[Tuple[int, int]]:
    """
    "1-40,45-60,70" -> [(1,40),(45,60),(70,70)]
    """
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
    if not ranges:
        return True  # إذا لم تحدد نطاق: صحح كل الأسئلة
    return any(a <= q <= b for a, b in ranges)

def rect_from_canvas_obj(obj: dict) -> Tuple[int, int, int, int]:
    # Fabric.js object
    x = int(obj.get("left", 0))
    y = int(obj.get("top", 0))
    w = int(obj.get("width", 0) * obj.get("scaleX", 1))
    h = int(obj.get("height", 0) * obj.get("scaleY", 1))
    return x, y, w, h

def clamp_rect(x, y, w, h, W, H):
    x = max(0, min(x, W-1))
    y = max(0, min(y, H-1))
    w = max(1, min(w, W-x))
    h = max(1, min(h, H-y))
    return x, y, w, h


# =========================
# OMR scoring (robust using ratios)
# =========================
def cell_fill_ratio(bin_img: np.ndarray) -> float:
    # نسبة البيكسلات البيضاء في الصورة الثنائية (THRESH_BINARY_INV)
    # كلما زادت = ظلل أكثر
    area = bin_img.shape[0] * bin_img.shape[1]
    if area <= 0:
        return 0.0
    filled = float(np.count_nonzero(bin_img))
    return filled / float(area)

def pick_one_by_ratio(ratios: List[Tuple[str, float]], min_fill: float, min_ratio: float):
    ratios = sorted(ratios, key=lambda x: x[1], reverse=True)
    top_c, top_r = ratios[0]
    second_r = ratios[1][1] if len(ratios) > 1 else 0.0

    if top_r < min_fill:
        return "?", "BLANK", top_r, second_r
    if second_r > 0 and (top_r / (second_r + 1e-9)) < min_ratio:
        return "!", "DOUBLE", top_r, second_r
    return top_c, "OK", top_r, second_r


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
    # canvas/template base size
    base_w: int
    base_h: int

    # ID ROI
    id_roi: Tuple[int, int, int, int]  # x,y,w,h
    id_digits: int
    id_rows: int  # غالبًا 10 (0..9)

    # blocks
    q_blocks: List[QBlock]

def resize_to_base(img_bgr: np.ndarray, base_w: int, base_h: int) -> np.ndarray:
    return cv2.resize(img_bgr, (base_w, base_h), interpolation=cv2.INTER_AREA)

def read_student_code(thr: np.ndarray, cfg: TemplateConfig,
                      min_fill: float = 0.20, min_ratio: float = 1.25) -> Tuple[str, List[dict]]:
    """
    يقرأ كود الطالب من شبكة (id_rows × id_digits)
    min_fill: نسبة تعبئة دنيا
    """
    x, y, w, h = cfg.id_roi
    H, W = thr.shape[:2]
    x, y, w, h = clamp_rect(x, y, w, h, W, H)
    roi = thr[y:y+h, x:x+w]

    rows = cfg.id_rows
    cols = cfg.id_digits
    ch = h // rows
    cw = w // cols

    digits = []
    debug = []
    for c in range(cols):
        ratios = []
        for r in range(rows):
            cell = roi[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            ratios.append((str(r), cell_fill_ratio(cell)))
        d, stt, top, sec = pick_one_by_ratio(ratios, min_fill=min_fill, min_ratio=min_ratio)
        debug.append({"digit_index": c, "status": stt, "top": top, "second": sec})
        digits.append("" if d in ["?","!"] else d)

    return "".join(digits), debug

def read_answers(thr: np.ndarray, cfg: TemplateConfig, choices: int,
                 min_fill: float = 0.12, min_ratio: float = 1.25) -> Tuple[Dict[int, Tuple[str, str]], List[dict]]:
    letters = "ABCDE"[:choices]
    out: Dict[int, Tuple[str, str]] = {}
    dbg: List[dict] = []

    H, W = thr.shape[:2]
    for b in cfg.q_blocks:
        x, y, w, h = clamp_rect(b.x, b.y, b.w, b.h, W, H)
        roi = thr[y:y+h, x:x+w]
        rows = max(1, b.rows)
        rh = h // rows
        cw = w // choices

        q = b.start_q
        for r in range(rows):
            if q > b.end_q:
                break
            ratios = []
            for c in range(choices):
                cell = roi[r*rh:(r+1)*rh, c*cw:(c+1)*cw]
                ratios.append((letters[c], cell_fill_ratio(cell)))
            a, stt, top, sec = pick_one_by_ratio(ratios, min_fill=min_fill, min_ratio=min_ratio)
            out[q] = (a, stt)
            dbg.append({"q": q, "status": stt, "top": top, "second": sec})
            q += 1

    return out, dbg


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="OMR BubbleSheet (Remark-style)", layout="wide")
st.title("✅ OMR BubbleSheet — تحديد بالماوس + تصحيح + Excel")

# Session state
if "template_img" not in st.session_state:
    st.session_state.template_img = None
if "cfg" not in st.session_state:
    st.session_state.cfg = None
if "qblocks_pending" not in st.session_state:
    st.session_state.qblocks_pending = []
if "id_roi" not in st.session_state:
    st.session_state.id_roi = None

left, right = st.columns([1.4, 1.0], gap="large")

with right:
    st.subheader("1) رفع نموذج الورقة (Template)")
    tpl_file = st.file_uploader("PDF/PNG/JPG (صفحة النموذج)", type=["pdf","png","jpg","jpeg"], key="tpl")

    st.divider()
    st.subheader("2) إعدادات الكود")
    id_digits = st.number_input("عدد خانات كود الطالب", 1, 20, 4, 1)
    id_rows = st.number_input("عدد صفوف أرقام الكود (عادة 10)", 5, 15, 10, 1)

    st.divider()
    st.subheader("3) إعداد بلوك الأسئلة قبل الحفظ")
    choices = st.radio("عدد الخيارات (A..)", [4, 5], horizontal=True)
    start_q = st.number_input("Start Q", 1, 500, 1, 1)
    end_q = st.number_input("End Q", 1, 500, 20, 1)
    rows_in_block = st.number_input("Rows داخل البلوك", 1, 300, 20, 1)

    st.caption("بعد ما ترسم مستطيل (Q Block) اضغط: إضافة بلوك")

    add_block = st.button("➕ إضافة بلوك من آخر مستطيل مرسوم", use_container_width=True)
    reset_all = st.button("♻️ Reset الكل", use_container_width=True)

    st.divider()
    st.subheader("4) التصحيح")
    roster_file = st.file_uploader("Roster (Excel/CSV) فيه student_code, student_name",
                                   type=["xlsx","xls","csv"], key="roster")
    key_file = st.file_uploader("Answer Key (PDF/صورة)", type=["pdf","png","jpg","jpeg"], key="key")
    sheets_file = st.file_uploader("أوراق الطلاب (PDF متعدد الصفحات أو صور)",
                                   type=["pdf","png","jpg","jpeg"], key="sheets")

    score_ranges_txt = st.text_input("نطاق التصحيح (مثال: 1-40 أو 1-70,1-25)", value="")
    strict_mode = st.checkbox("Strict: BLANK/DOUBLE = خطأ", value=True)

    go_grade = st.button("🚀 ابدأ التصحيح", type="primary", use_container_width=True)

if reset_all:
    st.session_state.template_img = None
    st.session_state.cfg = None
    st.session_state.qblocks_pending = []
    st.session_state.id_roi = None
    st.rerun()

with left:
    st.subheader("واجهة التحديد (ارسم مستطيل على الورقة)")

    if not tpl_file:
        st.info("ارفع Template أولًا من الجهة اليمنى.")
        st.stop()

    pages = load_pages(tpl_file.getvalue(), tpl_file.name)
    tpl = pages[0]
    st.session_state.template_img = tpl

    base_w, base_h = tpl.size  # PIL: (W,H)

    canvas_w = st.slider("Canvas width (كلما زاد أسرع/أبطأ حسب جهازك)", 700, 1800, min(1200, base_w), 50)
    scale = canvas_w / float(base_w)
    canvas_h = int(base_h * scale)

    draw_mode = st.radio("ماذا تحدد الآن؟", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], horizontal=True)

    # عرض canvas مع background image
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.12)",
        stroke_width=2,
        stroke_color="rgba(255, 0, 0, 0.9)",
        background_image=tpl.resize((canvas_w, canvas_h)),
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="rect",
        key="canvas",
    )

    last_rect = None
    if canvas_result and canvas_result.json_data:
        objs = canvas_result.json_data.get("objects", [])
        if objs:
            last_rect = rect_from_canvas_obj(objs[-1])

    st.write("آخر مستطيل:", last_rect if last_rect else "—")

    if add_block:
        if not last_rect:
            st.warning("ارسم مستطيل أولًا.")
        else:
            x, y, w, h = last_rect
            # رجّع الإحداثيات لحجم الصورة الأصلي
            X = int(x / scale); Y = int(y / scale); W = int(w / scale); H = int(h / scale)
            if draw_mode.startswith("ID"):
                st.session_state.id_roi = (X, Y, W, H)
                st.success(f"تم حفظ ID ROI: {(X,Y,W,H)}")
            else:
                st.session_state.qblocks_pending.append(QBlock(
                    x=X, y=Y, w=W, h=H,
                    start_q=int(start_q), end_q=int(end_q), rows=int(rows_in_block)
                ))
                st.success(f"تمت إضافة QBlock: ({start_q}..{end_q})")

    st.divider()
    st.subheader("المحفوظ حاليًا")
    st.write("ID ROI:", st.session_state.id_roi)
    st.write("Q Blocks:", [asdict(b) for b in st.session_state.qblocks_pending])

    if st.button("💾 حفظ القالب (Template Config)"):
        if not st.session_state.id_roi:
            st.error("لازم تحدد ID ROI أولًا.")
            st.stop()
        if not st.session_state.qblocks_pending:
            st.error("لازم تضيف على الأقل Q Block واحد.")
            st.stop()

        st.session_state.cfg = TemplateConfig(
            base_w=base_w,
            base_h=base_h,
            id_roi=st.session_state.id_roi,
            id_digits=int(id_digits),
            id_rows=int(id_rows),
            q_blocks=st.session_state.qblocks_pending
        )
        st.success("تم حفظ القالب ✅")

    if st.session_state.cfg:
        cfg_json = json.dumps(asdict(st.session_state.cfg), ensure_ascii=False, indent=2)
        st.download_button("تحميل config.json", cfg_json.encode("utf-8"), "config.json", "application/json")


# =========================
# Grading
# =========================
if go_grade:
    if not st.session_state.cfg:
        st.error("لازم تحفظ القالب أولًا (Template Config).")
        st.stop()
    if not (roster_file and key_file and sheets_file):
        st.error("ارفع Roster + Answer Key + أوراق الطلاب.")
        st.stop()

    # roster
    if roster_file.name.lower().endswith(("xlsx", "xls")):
        df_roster = pd.read_excel(roster_file)
    else:
        df_roster = pd.read_csv(roster_file)

    df_roster.columns = [c.strip().lower() for c in df_roster.columns]
    if "student_code" not in df_roster.columns or "student_name" not in df_roster.columns:
        st.error("ملف roster لازم يحتوي عمودين: student_code و student_name")
        st.stop()

    roster = dict(zip(df_roster["student_code"].astype(str), df_roster["student_name"].astype(str)))

    cfg: TemplateConfig = st.session_state.cfg

    # answer key
    key_pages = load_pages(key_file.getvalue(), key_file.name)
    key_bgr = resize_to_base(pil_to_cv(key_pages[0]), cfg.base_w, cfg.base_h)
    key_thr = preprocess(key_bgr)
    key_ans, key_dbg = read_answers(key_thr, cfg, choices)

    # student sheets
    sheet_pages = load_pages(sheets_file.getvalue(), sheets_file.name)
    score_ranges = parse_ranges(score_ranges_txt)

    results = []
    progress = st.progress(0)
    details = []

    for i, pg in enumerate(sheet_pages, start=1):
        img_bgr = resize_to_base(pil_to_cv(pg), cfg.base_w, cfg.base_h)
        thr = preprocess(img_bgr)

        code, code_dbg = read_student_code(thr, cfg)
        name = roster.get(str(code), "")

        stu_ans, stu_dbg = read_answers(thr, cfg, choices)

        score = 0
        total = 0

        for q, (ka, _) in key_ans.items():
            if not in_ranges(q, score_ranges):
                continue
            total += 1
            sa, stt = stu_ans.get(q, ("?", "BLANK"))

            if strict_mode and stt in ["BLANK", "DOUBLE"]:
                continue
            if sa == ka:
                score += 1

        results.append({
            "sheet_index": i,
            "student_code": code,
            "student_name": name,
            "score": score,
            "total": total
        })

        # اختياري: سجل التفاصيل لأكثر المشاكل
        details.append({
            "sheet_index": i,
            "student_code": code,
            "code_debug": code_dbg[:],
        })

        progress.progress(int(i / len(sheet_pages) * 100))

    out = pd.DataFrame(results)

    st.success("✅ تم التصحيح وإنشاء النتائج")
    st.dataframe(out, use_container_width=True)

    buf = io.BytesIO()
    out.to_excel(buf, index=False)
    st.download_button("⬇️ تحميل results.xlsx", buf.getvalue(), "results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
