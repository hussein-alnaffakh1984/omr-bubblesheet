# app.py  (Compatible with streamlit==1.40.0 + streamlit-drawable-canvas==0.9.3)

import io
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from streamlit_drawable_canvas import st_canvas


# =========================
# Data models
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
    id_roi: Tuple[int, int, int, int] = (0, 0, 10, 10)
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
# Helpers
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


def load_pages(file_bytes: bytes, filename: str, dpi: int = 200) -> List[Image.Image]:
    name = filename.lower()
    if name.endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        return [p.convert("RGB") for p in pages]
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return [img]


def resize_keep_ratio(img: Image.Image, target_w: int) -> Image.Image:
    w, h = img.size
    if w <= target_w:
        return img.convert("RGB")
    new_h = int(h * (target_w / w))
    return img.resize((target_w, new_h), Image.LANCZOS).convert("RGB")


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
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0

    if top_s < min_fill:
        return "?", "BLANK"
    if second_s > 0 and (top_s / (second_s + 1e-6)) < min_ratio:
        return "!", "DOUBLE"
    return top_c, "OK"


def crop(thr: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    x = max(0, x); y = max(0, y)
    return thr[y:y + h, x:x + w]


def read_student_code(thr: np.ndarray, cfg: TemplateConfig, min_fill=250, min_ratio=1.25) -> str:
    x, y, w, h = cfg.id_roi
    roi = crop(thr, (x, y, w, h))
    if roi.size == 0:
        return ""

    rows, cols = cfg.id_rows, cfg.id_digits
    ch = max(1, roi.shape[0] // rows)
    cw = max(1, roi.shape[1] // cols)

    digits = []
    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            scores.append((str(r), score_cell(cell)))
        d, _ = pick_one(scores, min_fill, min_ratio)
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
                cell = roi[r * rh:(r + 1) * rh, c * cw:(c + 1) * cw]
                scores.append((letters[c], score_cell(cell)))

            a, stt = pick_one(scores, min_fill, min_ratio)
            out[q] = (a, stt)
            q += 1

    return out


def normalize_code(code: str, digits: int) -> str:
    code = (code or "").strip()
    code2 = re.sub(r"\D+", "", code)
    if digits > 0 and code2:
        code2 = code2.zfill(digits)
    return code2


def load_roster(file) -> Dict[str, str]:
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "student_code" not in df.columns or "student_name" not in df.columns:
        raise ValueError("الروستر يجب أن يحتوي: student_code و student_name")

    codes = df["student_code"].astype(str).str.strip()
    names = df["student_name"].astype(str).str.strip()
    return dict(zip(codes, names))


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="OMR Remark-Style", layout="wide")
st.title("✅ OMR Bubble Sheet — Remark-Style (Streamlit 1.40 Compatible)")

# init session state
if "cfg" not in st.session_state:
    st.session_state["cfg"] = TemplateConfig(q_blocks=[])
if "template_bytes" not in st.session_state:
    st.session_state["template_bytes"] = None
    st.session_state["template_name"] = None
if "canvas_width_fixed" not in st.session_state:
    st.session_state["canvas_width_fixed"] = 1200  # fixed width after set


# Sidebar
st.sidebar.header("إعدادات")
choices = st.sidebar.radio("عدد الخيارات", [4, 5], horizontal=True)
dpi = st.sidebar.selectbox("DPI للـ PDF", [150, 200, 250], index=1)

st.sidebar.divider()
min_fill_id = st.sidebar.slider("min_fill (ID)", 50, 800, 250, 10)
min_fill_q = st.sidebar.slider("min_fill (Q)", 50, 800, 180, 10)
min_ratio = st.sidebar.slider("min_ratio", 1.05, 3.0, 1.25, 0.05)
strict = st.sidebar.checkbox("وضع صارم", True)


# Layout
colL, colR = st.columns([1.35, 1.0], gap="large")

with colR:
    st.header("1) رفع Template")
    template_file = st.file_uploader("PDF/PNG/JPG (صفحة النموذج)", type=["pdf", "png", "jpg", "jpeg"])
    if template_file is not None:
        st.session_state["template_bytes"] = template_file.getvalue()
        st.session_state["template_name"] = template_file.name

    if st.session_state["template_bytes"] is None:
        st.info("ارفع Template أولاً.")
        st.stop()

    cfg: TemplateConfig = st.session_state["cfg"]

    st.subheader("2) إعدادات كود الطالب")
    cfg.id_digits = int(st.number_input("عدد خانات الكود", 1, 12, int(cfg.id_digits)))
    cfg.id_rows = int(st.number_input("عدد صفوف أرقام الكود", 5, 12, int(cfg.id_rows)))

    st.subheader("3) إعداد بلوك الأسئلة")
    start_q = int(st.number_input("Start Q", 1, 500, 1))
    end_q = int(st.number_input("End Q", 1, 500, 60))
    rows_in_block = int(st.number_input("Rows داخل البلوك", 1, 200, 20))

    st.subheader("تثبيت عرض الكانفاس (مهم)")
    new_w = st.slider("Canvas Width", 700, 1800, int(st.session_state["canvas_width_fixed"]), 50)
    if st.button("📌 تثبيت العرض (ثم ارسم)"):
        st.session_state["canvas_width_fixed"] = int(new_w)
        st.success("تم تثبيت عرض الكانفاس. الآن ارسم بدون تغيير العرض.")
        st.rerun()


with colL:
    st.header("واجهة التحديد (ارسم مستطيل)")

    pages = load_pages(st.session_state["template_bytes"], st.session_state["template_name"], dpi=dpi)
    base_img = pages[0].convert("RGB")
    base_w, base_h = base_img.size

    canvas_w = int(st.session_state["canvas_width_fixed"])
    bg_img = resize_keep_ratio(base_img, canvas_w)
    bg_w, bg_h = bg_img.size

    # IMPORTANT: pass background as numpy array (fixes white/blank on Streamlit Cloud)
    bg_np = np.array(bg_img)

    scale = base_w / bg_w
    st.caption(f"Original: {base_w}×{base_h} | Canvas: {bg_w}×{bg_h} | scale={scale:.4f}")

    mode = st.radio("تحدد الآن:", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], horizontal=True)

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.20)",
        stroke_width=3,
        stroke_color="red",
        background_color="rgba(0,0,0,0)",
        background_image=bg_np,   # ✅ numpy not PIL
        update_streamlit=True,
        height=bg_h,
        width=bg_w,
        drawing_mode="rect",
        key="omr_canvas_fixed"
    )

    b1, b2, b3, b4 = st.columns(4)
    clear_btn = b1.button("🧹 Clear")
    reset_btn = b2.button("♻️ Reset")
    save_id_btn = b3.button("💾 حفظ ID ROI")
    add_blk_btn = b4.button("➕ إضافة Q Block")

    if clear_btn:
        st.session_state["omr_canvas_fixed"] = None
        st.rerun()

    if reset_btn:
        st.session_state["cfg"] = TemplateConfig(q_blocks=[])
        st.session_state["omr_canvas_fixed"] = None
        st.rerun()

    def get_last_rect():
        if not canvas_result or not canvas_result.json_data:
            return None
        objs = canvas_result.json_data.get("objects", [])
        if not objs:
            return None
        obj = objs[-1]
        x = int(obj.get("left", 0))
        y = int(obj.get("top", 0))
        w = int(obj.get("width", 0))
        h = int(obj.get("height", 0))
        X = int(round(x * scale))
        Y = int(round(y * scale))
        W = int(round(w * scale))
        H = int(round(h * scale))
        return (X, Y, W, H)

    last_rect = get_last_rect()

    if save_id_btn:
        if not last_rect:
            st.error("ارسم مستطيل على منطقة كود الطالب أولاً.")
        else:
            cfg = st.session_state["cfg"]
            cfg.id_roi = last_rect
            st.session_state["cfg"] = cfg
            st.success(f"تم حفظ ID ROI: {cfg.id_roi}")

    if add_blk_btn:
        if not last_rect:
            st.error("ارسم مستطيل على بلوك الأسئلة أولاً.")
        else:
            cfg = st.session_state["cfg"]
            qb = QBlock(
                x=last_rect[0], y=last_rect[1], w=last_rect[2], h=last_rect[3],
                start_q=start_q, end_q=end_q, rows=rows_in_block
            )
            cfg.q_blocks = (cfg.q_blocks or []) + [qb]
            st.session_state["cfg"] = cfg
            st.success(f"تمت إضافة Q Block #{len(cfg.q_blocks)}")

    cfg = st.session_state["cfg"]
    st.subheader("Config الحالي")
    st.json(cfg.to_dict())

    cfg_bytes = json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("⬇️ تحميل config.json", cfg_bytes, file_name="config.json", mime="application/json")


st.divider()
st.header("2) التصحيح")

c1, c2, c3 = st.columns(3)
roster_file = c1.file_uploader("Roster (student_code, student_name)", type=["xlsx", "xls", "csv"])
key_file = c2.file_uploader("Answer Key", type=["pdf", "png", "jpg", "jpeg"])
sheets_file = c3.file_uploader("Student Sheets", type=["pdf", "png", "jpg", "jpeg"])

range_txt = st.text_input("نطاق الأسئلة (مثال: 1-60 أو 1-40,45-60)", "1-60")
ranges = parse_ranges(range_txt)

run_btn = st.button("🚀 ابدأ التصحيح", type="primary")

if run_btn:
    cfg: TemplateConfig = st.session_state["cfg"]

    if roster_file is None or key_file is None or sheets_file is None:
        st.error("ارفع Roster + Answer Key + أوراق الطلاب.")
        st.stop()

    if not ranges:
        st.error("اكتب نطاق أسئلة صحيح.")
        st.stop()

    if cfg.id_roi[2] <= 10 or cfg.id_roi[3] <= 10:
        st.error("حدد ID ROI بشكل صحيح.")
        st.stop()

    if not cfg.q_blocks:
        st.error("أضف على الأقل Q Block واحد.")
        st.stop()

    roster = load_roster(roster_file)

    key_pages = load_pages(key_file.getvalue(), key_file.name, dpi=dpi)
    key_thr = preprocess(pil_to_cv(key_pages[0]))
    key_ans = read_answers(key_thr, cfg, choices, min_fill=min_fill_q, min_ratio=min_ratio)

    pages = load_pages(sheets_file.getvalue(), sheets_file.name, dpi=dpi)

    results = []
    prog = st.progress(0)

    for i, pg in enumerate(pages, 1):
        thr = preprocess(pil_to_cv(pg))

        raw_code = read_student_code(thr, cfg, min_fill=min_fill_id, min_ratio=min_ratio)
        code = normalize_code(raw_code, cfg.id_digits)
        name = roster.get(code, "")

        stu_ans = read_answers(thr, cfg, choices, min_fill=min_fill_q, min_ratio=min_ratio)

        score = 0
        for q, (ka, _) in key_ans.items():
            if not in_ranges(q, ranges):
                continue
            sa, sa_state = stu_ans.get(q, ("?", "MISSING"))

            if strict and sa_state != "OK":
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

    df_out = pd.DataFrame(results)
    st.dataframe(df_out, use_container_width=True)

    buf = io.BytesIO()
    df_out.to_excel(buf, index=False)
    st.download_button("⬇️ تحميل النتائج Excel", buf.getvalue(), "results.xlsx")

    st.success("تم ✅")
