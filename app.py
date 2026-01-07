# app.py
# OMR Bubble Sheet (Remark-style) — Streamlit Community Cloud
# ✅ رسم مستطيل يدوي لتحديد: (1) منطقة كود الطالب ID ROI  (2) بلوكات الأسئلة Q Blocks
# ✅ يدعم 4 أو 5 خيارات
# ✅ يدعم نطاقات (نظري + عملي) أو نطاق واحد
# ✅ يقرأ: sheet_index + student_code + student_name + score  ويصدر Excel

import io
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from streamlit_drawable_canvas import st_canvas


# =========================
# Config models
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
    # original-image coordinates (not resized)
    id_roi: Tuple[int, int, int, int] = (0, 0, 10, 10)  # x,y,w,h
    id_digits: int = 4
    id_rows: int = 10
    q_blocks: List[QBlock] = None

    def to_dict(self):
        d = asdict(self)
        d["q_blocks"] = [asdict(b) for b in (self.q_blocks or [])]
        return d

    @staticmethod
    def from_dict(d):
        cfg = TemplateConfig()
        cfg.id_roi = tuple(d.get("id_roi", cfg.id_roi))
        cfg.id_digits = int(d.get("id_digits", cfg.id_digits))
        cfg.id_rows = int(d.get("id_rows", cfg.id_rows))
        qb = d.get("q_blocks", [])
        cfg.q_blocks = [QBlock(**b) for b in qb]
        return cfg


# =========================
# Helpers: parsing ranges
# =========================
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
    if not ranges:
        return True
    return any(a <= q <= b for a, b in ranges)


# =========================
# Helpers: PDF/Image loading
# =========================
def load_pages(file_bytes: bytes, filename: str, dpi: int = 200) -> List[Image.Image]:
    name = filename.lower()
    if name.endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        return [p.convert("RGB") for p in pages]
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return [img]

def resize_keep_ratio(img: Image.Image, target_w: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w <= target_w:
        return img
    new_h = int(h * (target_w / w))
    return img.resize((target_w, new_h), Image.LANCZOS)

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
    # scores: [(label, fill_pixels), ...]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0

    if top_s < min_fill:
        return "?", "BLANK"
    if second_s > 0 and (top_s / (second_s + 1e-6)) < min_ratio:
        return "!", "DOUBLE"
    return top_c, "OK"


# =========================
# OMR readers using config
# =========================
def crop(thr: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    x = max(0, x); y = max(0, y)
    return thr[y:y+h, x:x+w]

def read_student_code(thr: np.ndarray, cfg: TemplateConfig, min_fill=250, min_ratio=1.25) -> str:
    x, y, w, h = cfg.id_roi
    roi = crop(thr, (x, y, w, h))

    rows, cols = cfg.id_rows, cfg.id_digits
    if rows <= 0 or cols <= 0 or roi.size == 0:
        return ""

    ch = max(1, roi.shape[0] // rows)
    cw = max(1, roi.shape[1] // cols)

    digits = []
    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            scores.append((str(r), score_cell(cell)))
        d, stt = pick_one(scores, min_fill, min_ratio)
        digits.append("" if d in ["?", "!"] else d)
    return "".join(digits)

def read_answers(thr: np.ndarray, cfg: TemplateConfig, choices: int, min_fill=180, min_ratio=1.25) -> Dict[int, Tuple[str, str]]:
    letters = "ABCDE"[:choices]
    out = {}

    for blk in (cfg.q_blocks or []):
        roi = crop(thr, (blk.x, blk.y, blk.w, blk.h))
        if roi.size == 0:
            continue

        rows = max(1, int(blk.rows))
        rh = max(1, roi.shape[0] // rows)
        cw = max(1, roi.shape[1] // choices)

        q = int(blk.start_q)
        end_q = int(blk.end_q)

        for r in range(rows):
            if q > end_q:
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
# Streamlit App
# =========================
st.set_page_config(page_title="OMR Bubble Sheet (Remark-Style)", layout="wide")
st.title("✅ تصحيح ببل شيت — Remark-Style (تحديد يدوي + تصحيح)")

# -------------------------
# State init
# -------------------------
if "cfg" not in st.session_state:
    st.session_state["cfg"] = TemplateConfig(q_blocks=[])
if "template_bytes" not in st.session_state:
    st.session_state["template_bytes"] = None
    st.session_state["template_name"] = None
if "template_base_size" not in st.session_state:
    st.session_state["template_base_size"] = None  # (w,h)
if "last_scale" not in st.session_state:
    st.session_state["last_scale"] = 1.0


# -------------------------
# Sidebar: global settings
# -------------------------
st.sidebar.header("إعدادات عامة")
choices = st.sidebar.radio("عدد الخيارات", [4, 5], horizontal=True)
dpi = st.sidebar.selectbox("DPI للـ PDF (أقل = أسرع)", [150, 200, 250], index=1)

st.sidebar.divider()
st.sidebar.subheader("حساسية القراءة (تعديل عند الخطأ)")
min_fill_id = st.sidebar.slider("min_fill (كود الطالب)", 50, 800, 250, 10)
min_fill_q  = st.sidebar.slider("min_fill (الأسئلة)",  50, 800, 180, 10)
min_ratio   = st.sidebar.slider("min_ratio (تمييز مزدوج)", 1.05, 3.0, 1.25, 0.05)

st.sidebar.divider()
strict = st.sidebar.checkbox("وضع صارم: BLANK/DOUBLE = خطأ", True)


# =========================
# 1) TEMPLATE + DRAWING
# =========================
colL, colR = st.columns([1.35, 1.0], gap="large")

with colR:
    st.header("1) رفع نموذج الورقة (Template)")
    template_file = st.file_uploader("PDF/PNG/JPG (صفحة النموذج)", type=["pdf", "png", "jpg", "jpeg"])

    if template_file is not None:
        st.session_state["template_bytes"] = template_file.getvalue()
        st.session_state["template_name"] = template_file.name

    if st.session_state["template_bytes"] is None:
        st.info("⬅️ ارفع ملف النموذج حتى يظهر على اليسار وتبدأ التحديد.")
        st.stop()

    st.subheader("2) إعدادات الكود")
    cfg: TemplateConfig = st.session_state["cfg"]
    cfg.id_digits = int(st.number_input("عدد خانات كود الطالب", 1, 12, int(cfg.id_digits)))
    cfg.id_rows = int(st.number_input("عدد صفوف أرقام الكود (عادة 10)", 5, 12, int(cfg.id_rows)))

    st.subheader("3) إعداد بلوكات الأسئلة قبل الحفظ")
    start_q = int(st.number_input("Start Q", 1, 500, 1))
    end_q   = int(st.number_input("End Q", 1, 500, 60))
    rows_in_block = int(st.number_input("Rows داخل البلوك", 1, 200, 20))

    st.caption("💡 ارسم مستطيل على الصورة (يسار) ثم اضغط زر الإضافة المناسب.")


with colL:
    st.header("واجهة التحديد (ارسم مستطيل على الورقة)")

    # Load base image (first page)
    pages = load_pages(st.session_state["template_bytes"], st.session_state["template_name"], dpi=dpi)
    base_img = pages[0].convert("RGB")
    base_w, base_h = base_img.size
    st.session_state["template_base_size"] = (base_w, base_h)

    canvas_w = st.slider("Canvas width (غيّره حسب جهازك)", 700, 1800, 1250, 50)

    # Resize for canvas
    bg_img = resize_keep_ratio(base_img, canvas_w)
    bg_w, bg_h = bg_img.size
    scale = base_w / bg_w
    st.session_state["last_scale"] = scale

    st.caption(f"حجم الصورة: الأصل {base_w}×{base_h} | العرض على الكانفاس {bg_w}×{bg_h} | scale={scale:.4f}")

    mode = st.radio("ماذا تحدد الآن؟", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], horizontal=True)

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.20)",
        stroke_width=3,
        stroke_color="red",
        background_color="rgba(0,0,0,0)",
        background_image=bg_img,
        update_streamlit=True,
        height=bg_h,
        width=bg_w,
        drawing_mode="rect",
        key="omr_canvas"
    )

    # buttons row
    b1, b2, b3, b4 = st.columns([1,1,1,1])
    with b1:
        clear_canvas = st.button("🧹 Clear Canvas")
    with b2:
        reset_all = st.button("♻️ Reset الكل (ID + Blocks)")
    with b3:
        add_id = st.button("➕ حفظ ID ROI")
    with b4:
        add_blk = st.button("➕ إضافة Q Block")

    if clear_canvas:
        st.session_state["omr_canvas"] = None
        st.rerun()

    if reset_all:
        st.session_state["cfg"] = TemplateConfig(q_blocks=[])
        st.session_state["omr_canvas"] = None
        st.rerun()

    # extract last drawn rect
    def get_last_rect():
        if not canvas_result or not canvas_result.json_data:
            return None
        objs = canvas_result.json_data.get("objects", [])
        if not objs:
            return None
        obj = objs[-1]
        # streamlit_drawable_canvas gives: left, top, width, height
        x = int(obj.get("left", 0))
        y = int(obj.get("top", 0))
        w = int(obj.get("width", 0))
        h = int(obj.get("height", 0))
        # map to original coordinates
        X = int(round(x * scale))
        Y = int(round(y * scale))
        W = int(round(w * scale))
        H = int(round(h * scale))
        return (X, Y, W, H)

    last_rect = get_last_rect()

    if add_id:
        if not last_rect:
            st.error("ارسم مستطيل أولاً على منطقة كود الطالب.")
        else:
            cfg = st.session_state["cfg"]
            cfg.id_roi = last_rect
            st.session_state["cfg"] = cfg
            st.success(f"تم حفظ ID ROI: {cfg.id_roi}")

    if add_blk:
        if not last_rect:
            st.error("ارسم مستطيل أولاً على بلوك الأسئلة.")
        else:
            cfg = st.session_state["cfg"]
            qb = QBlock(
                x=last_rect[0], y=last_rect[1], w=last_rect[2], h=last_rect[3],
                start_q=start_q, end_q=end_q, rows=rows_in_block
            )
            cfg.q_blocks = (cfg.q_blocks or []) + [qb]
            st.session_state["cfg"] = cfg
            st.success(f"تمت إضافة Q Block #{len(cfg.q_blocks)}")

    # show current config
    cfg = st.session_state["cfg"]
    st.subheader("✅ الإعدادات الحالية (Config)")
    st.json(cfg.to_dict())

    # export/import config json
    c1, c2 = st.columns([1,1])
    with c1:
        cfg_bytes = json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("⬇️ تحميل config.json", cfg_bytes, file_name="config.json", mime="application/json")
    with c2:
        cfg_upload = st.file_uploader("تحميل config.json (اختياري)", type=["json"])
        if cfg_upload is not None:
            try:
                d = json.loads(cfg_upload.getvalue().decode("utf-8"))
                st.session_state["cfg"] = TemplateConfig.from_dict(d)
                st.success("تم تحميل config.json بنجاح")
                st.rerun()
            except Exception as e:
                st.error(f"فشل تحميل config.json: {e}")


st.divider()


# =========================
# 2) GRADING
# =========================
st.header("✅ التصحيح الفعلي")

g1, g2, g3 = st.columns([1,1,1], gap="large")

with g1:
    st.subheader("A) ملف الطلاب (Roster)")
    roster_file = st.file_uploader("Excel/CSV: student_code, student_name", type=["xlsx", "xls", "csv"], key="roster")
with g2:
    st.subheader("B) Answer Key")
    key_file = st.file_uploader("PDF صفحة واحدة أو صورة", type=["pdf", "png", "jpg", "jpeg"], key="key")
with g3:
    st.subheader("C) أوراق الطلاب")
    sheets_file = st.file_uploader("PDF متعدد الصفحات أو صور", type=["pdf", "png", "jpg", "jpeg"], key="sheets")

st.subheader("نطاق الأسئلة التي تُحسب في الدرجة")
range_mode = st.radio("طريقة النطاق", ["نطاق واحد", "نظري + عملي"], horizontal=True)

if range_mode == "نطاق واحد":
    all_txt = st.text_input("النطاق (مثال: 1-70 أو 1-40, 45-60)", "1-60")
    theory_ranges = parse_ranges(all_txt)
    practical_ranges = []
else:
    theory_txt = st.text_input("نطاق النظري", "1-60")
    practical_txt = st.text_input("نطاق العملي (اختياري)", "")
    theory_ranges = parse_ranges(theory_txt)
    practical_ranges = parse_ranges(practical_txt)

st.caption("ملاحظة: إذا تركت النطاق فارغًا لن يحسب شيء. اكتب نطاقًا صحيحًا.")

run_btn = st.button("🚀 ابدأ التصحيح الآن", type="primary")

def load_roster(file) -> Dict[str, str]:
    if file is None:
        return {}
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [str(c).strip().lower() for c in df.columns]
    # required columns
    if "student_code" not in df.columns or "student_name" not in df.columns:
        raise ValueError("الملف يجب أن يحتوي عمودين: student_code و student_name")
    codes = df["student_code"].astype(str).str.strip()
    names = df["student_name"].astype(str).str.strip()
    return dict(zip(codes, names))

def normalize_code(code: str, digits: int) -> str:
    code = (code or "").strip()
    # keep digits only
    code2 = re.sub(r"\D+", "", code)
    if digits > 0 and code2:
        code2 = code2.zfill(digits)
    return code2

if run_btn:
    cfg: TemplateConfig = st.session_state["cfg"]

    # quick validation
    if cfg.id_roi[2] <= 10 or cfg.id_roi[3] <= 10:
        st.error("حدد ID ROI بشكل صحيح أولًا (حجم أكبر من 10×10).")
        st.stop()
    if not cfg.q_blocks or len(cfg.q_blocks) == 0:
        st.error("أضف على الأقل Q Block واحد للأسئلة.")
        st.stop()
    if roster_file is None or key_file is None or sheets_file is None:
        st.error("ارفع Roster + Answer Key + أوراق الطلاب.")
        st.stop()
    if not theory_ranges and not practical_ranges:
        st.error("اكتب نطاق أسئلة ليتم حساب الدرجة.")
        st.stop()

    # load roster
    try:
        roster = load_roster(roster_file)
        st.success(f"تم تحميل roster: {len(roster)} طالب")
    except Exception as e:
        st.error(str(e))
        st.stop()

    # read answer key
    key_pages = load_pages(key_file.getvalue(), key_file.name, dpi=dpi)
    key_thr = preprocess(pil_to_cv(key_pages[0]))
    key_ans = read_answers(key_thr, cfg, choices=choices, min_fill=min_fill_q, min_ratio=min_ratio)

    if len(key_ans) == 0:
        st.error("لم أستطع قراءة Answer Key (تحقق من Q Blocks أو الحساسية min_fill).")
        st.stop()

    # load student pages
    pages = load_pages(sheets_file.getvalue(), sheets_file.name, dpi=dpi)
    st.info(f"عدد صفحات الطلاب: {len(pages)}")

    prog = st.progress(0)
    results = []
    debug_rows = []

    for i, pg in enumerate(pages, 1):
        thr = preprocess(pil_to_cv(pg))

        raw_code = read_student_code(thr, cfg, min_fill=min_fill_id, min_ratio=min_ratio)
        code = normalize_code(raw_code, cfg.id_digits)
        name = roster.get(code, "")

        stu_ans = read_answers(thr, cfg, choices=choices, min_fill=min_fill_q, min_ratio=min_ratio)

        # scoring
        score = 0
        total_counted = 0

        for q, (ka, ka_state) in key_ans.items():
            sa, sa_state = stu_ans.get(q, ("?", "MISSING"))

            count_this = False
            if theory_ranges and in_ranges(q, theory_ranges):
                count_this = True
            if practical_ranges and in_ranges(q, practical_ranges):
                count_this = True

            if not count_this:
                continue

            total_counted += 1

            if strict:
                # strict: only OK answers count
                if sa_state != "OK":
                    continue
            # compare
            if sa == ka:
                score += 1

        results.append({
            "sheet_index": i,
            "student_code": code,
            "student_name": name,
            "score": score,
        })

        # optional debug: show unread codes
        if code == "" or name == "":
            debug_rows.append({
                "sheet_index": i,
                "raw_code": raw_code,
                "normalized_code": code,
                "name_found": bool(name),
            })

        prog.progress(int(i / len(pages) * 100))

    out_df = pd.DataFrame(results)

    st.subheader("📌 النتائج")
    st.dataframe(out_df, use_container_width=True)

    # export excel
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="results")
        if debug_rows:
            pd.DataFrame(debug_rows).to_excel(writer, index=False, sheet_name="debug")
    st.download_button("⬇️ تحميل النتائج Excel", buf.getvalue(), "results.xlsx")

    st.success("تمت العملية ✅")

    if debug_rows:
        st.warning("بعض الأكواد/الأسماء لم تُقرأ أو لم تُطابق roster. راجع ورقة debug داخل Excel.")
