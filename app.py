import io
import json
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw

# ✅ موجود عندك حسب اللوج
from streamlit_image_coordinates import streamlit_image_coordinates


# ----------------------------
# Data models
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
    id_roi: Tuple[int, int, int, int] = (0, 0, 0, 0)
    id_digits: int = 4
    id_rows: int = 10
    q_blocks: List[QBlock] = None

    def to_dict(self):
        return {
            "id_roi": list(self.id_roi),
            "id_digits": self.id_digits,
            "id_rows": self.id_rows,
            "q_blocks": [
                dict(x=b.x, y=b.y, w=b.w, h=b.h, start_q=b.start_q, end_q=b.end_q, rows=b.rows)
                for b in (self.q_blocks or [])
            ],
        }

    @staticmethod
    def from_dict(d):
        cfg = TemplateConfig()
        cfg.id_roi = tuple(d.get("id_roi", [0, 0, 0, 0]))
        cfg.id_digits = int(d.get("id_digits", 4))
        cfg.id_rows = int(d.get("id_rows", 10))
        cfg.q_blocks = [
            QBlock(
                x=int(b["x"]), y=int(b["y"]), w=int(b["w"]), h=int(b["h"]),
                start_q=int(b["start_q"]), end_q=int(b["end_q"]), rows=int(b["rows"])
            )
            for b in d.get("q_blocks", [])
        ]
        return cfg


# ----------------------------
# Helpers
# ----------------------------
def load_pages(file_bytes: bytes, name: str) -> List[Image.Image]:
    name = name.lower()
    if name.endswith(".pdf"):
        return convert_from_bytes(file_bytes)
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


def pick_one(scores, min_fill, min_ratio):
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


def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


# ----------------------------
# OMR readers
# ----------------------------
def read_student_code(thr: np.ndarray, cfg: TemplateConfig, min_fill=180, min_ratio=1.25) -> str:
    x, y, w, h = cfg.id_roi
    H, W = thr.shape[:2]
    x, y, w, h = clamp_roi(x, y, w, h, W, H)

    roi = thr[y:y+h, x:x+w]
    rows, cols = cfg.id_rows, cfg.id_digits
    ch, cw = max(1, h // rows), max(1, w // cols)

    digits = []
    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            scores.append((str(r), score_cell(cell)))
        d, stt = pick_one(scores, min_fill, min_ratio)
        digits.append("0" if d in ["?","!"] else d)

    code = "".join(digits)
    return code


def read_answers(thr: np.ndarray, blocks: List[QBlock], choices: int, min_fill=120, min_ratio=1.25) -> Dict[int, Tuple[str, str]]:
    letters = "ABCDE"[:choices]
    out: Dict[int, Tuple[str, str]] = {}

    H, W = thr.shape[:2]
    for b in blocks:
        x, y, w, h = clamp_roi(b.x, b.y, b.w, b.h, W, H)
        roi = thr[y:y+h, x:x+w]

        rows = b.rows
        rh = max(1, h // rows)
        cw = max(1, w // choices)

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


# ----------------------------
# ROI Selection (Remark-style: two clicks)
# ----------------------------
def draw_overlays(img: Image.Image, cfg: TemplateConfig) -> Image.Image:
    out = img.copy()
    dr = ImageDraw.Draw(out)

    # ID ROI
    x, y, w, h = cfg.id_roi
    if w > 0 and h > 0:
        dr.rectangle([x, y, x+w, y+h], outline="red", width=4)

    # Q blocks
    for i, b in enumerate(cfg.q_blocks or []):
        dr.rectangle([b.x, b.y, b.x+b.w, b.y+b.h], outline="lime", width=4)
        dr.text((b.x+6, b.y+6), f"Q{i+1}: {b.start_q}-{b.end_q}", fill="lime")

    return out


def rect_from_two_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    x = int(min(x1, x2))
    y = int(min(y1, y2))
    w = int(abs(x2 - x1))
    h = int(abs(y2 - y1))
    return x, y, w, h


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="OMR Bubble Sheet (Remark-style)", layout="wide")
st.title("✅ OMR Bubble Sheet — واجهة Remark (نقرتين لتحديد المستطيل)")

if "cfg" not in st.session_state:
    st.session_state.cfg = TemplateConfig(id_roi=(0, 0, 0, 0), id_digits=4, id_rows=10, q_blocks=[])

if "picks" not in st.session_state:
    st.session_state.picks = []  # list of (x,y) in ORIGINAL IMAGE coords

cfg: TemplateConfig = st.session_state.cfg

# --- Sidebar: files
with st.sidebar:
    st.header("1) رفع نموذج الورقة (Template)")
    tpl_file = st.file_uploader("PDF/PNG/JPG (صفحة واحدة)", type=["pdf", "png", "jpg", "jpeg"])
    st.divider()

    st.header("2) إعدادات الكود")
    cfg.id_digits = st.number_input("عدد خانات كود الطالب", 1, 12, int(cfg.id_digits), 1)
    cfg.id_rows = st.number_input("عدد صفوف أرقام الكود (عادة 10)", 5, 12, int(cfg.id_rows), 1)

    st.divider()
    st.header("3) إعدادات الاختيارات والأسئلة")
    choices = st.radio("عدد الخيارات", [4, 5], horizontal=True)

    st.caption("✅ إضافة بلوك أسئلة: حدده بنقرتين ثم اضغط (أضف بلوك)")
    start_q = st.number_input("Start Q", 1, 500, 1, 1)
    end_q = st.number_input("End Q", 1, 500, 20, 1)
    rows_in_block = st.number_input("Rows داخل البلوك", 1, 200, 20, 1)

    st.divider()
    st.header("4) مفاتيح التصحيح والطلاب")
    roster_file = st.file_uploader("Roster Excel: student_code, student_name", type=["xlsx", "xls", "csv"])
    key_file = st.file_uploader("Answer Key (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"])
    sheets_file = st.file_uploader("أوراق الطلاب (PDF متعدد الصفحات أو صور)", type=["pdf", "png", "jpg", "jpeg"])

    st.divider()
    st.header("5) حساسية القراءة")
    min_fill_id = st.slider("min_fill (ID)", 50, 1000, 180, 10)
    min_fill_q = st.slider("min_fill (Q)", 50, 1000, 120, 10)
    min_ratio = st.slider("min_ratio", 1.05, 2.50, 1.25, 0.05)

# --- Main area
colA, colB = st.columns([1.6, 1.0])

with colA:
    st.subheader("واجهة التحديد (نقرتين فقط)")
    if not tpl_file:
        st.info("ارفع Template أولاً من الشريط الجانبي.")
        st.stop()

    tpl_pages = load_pages(tpl_file.getvalue(), tpl_file.name)
    tpl_img = tpl_pages[0].convert("RGB")
    W0, H0 = tpl_img.size

    # Resize for display (to avoid huge image)
    max_display_w = 1200
    display_w = min(max_display_w, W0)
    scale = display_w / W0
    display_h = int(H0 * scale)

    # overlay rectangles on ORIGINAL then resize for display
    overlay = draw_overlays(tpl_img, cfg)
    overlay_disp = overlay.resize((display_w, display_h), Image.LANCZOS)

    st.caption(f"Template size: {W0}×{H0} | عرض العرض: {display_w}px | scale={scale:.4f}")

    mode = st.radio("ماذا نحدد الآن؟", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], horizontal=True)

    # click picker (returns coords in DISPLAY space) -> convert to ORIGINAL
    picked = streamlit_image_coordinates(overlay_disp, key="picker")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("🧹 مسح آخر نقطة"):
            if st.session_state.picks:
                st.session_state.picks.pop()
    with c2:
        if st.button("♻️ Reset النقاط"):
            st.session_state.picks = []
    with c3:
        if st.button("🗑️ حذف كل البلوكات"):
            cfg.q_blocks = []
            cfg.id_roi = (0, 0, 0, 0)
            st.session_state.picks = []

    if picked and "x" in picked and "y" in picked:
        # Convert display -> original
        ox = int(picked["x"] / scale)
        oy = int(picked["y"] / scale)
        st.session_state.picks.append((ox, oy))

    st.write("النقاط الحالية (بالإحداثيات الأصلية):", st.session_state.picks[-2:])

    if len(st.session_state.picks) >= 2:
        p1 = st.session_state.picks[-2]
        p2 = st.session_state.picks[-1]
        x, y, w, h = rect_from_two_points(p1, p2)
        x, y, w, h = clamp_roi(x, y, w, h, W0, H0)

        st.success(f"المستطيل المقترح: x={x}, y={y}, w={w}, h={h}")

        if mode.startswith("ID ROI"):
            if st.button("✅ حفظ ID ROI"):
                cfg.id_roi = (x, y, w, h)
                st.toast("تم حفظ منطقة كود الطالب")
        else:
            if st.button("✅ أضف بلوك أسئلة"):
                if end_q < start_q:
                    st.error("End Q يجب أن يكون أكبر أو يساوي Start Q")
                else:
                    cfg.q_blocks.append(QBlock(x=x, y=y, w=w, h=h, start_q=int(start_q), end_q=int(end_q), rows=int(rows_in_block)))
                    st.toast("تمت إضافة بلوك أسئلة")

with colB:
    st.subheader("إعدادات القالب (Template Config)")
    st.json(cfg.to_dict())

    st.download_button(
        "⬇️ تحميل config.json",
        data=json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="config.json",
        mime="application/json",
    )

    up_cfg = st.file_uploader("رفع config.json (اختياري)", type=["json"])
    if up_cfg:
        try:
            cfg2 = json.loads(up_cfg.getvalue().decode("utf-8"))
            st.session_state.cfg = TemplateConfig.from_dict(cfg2)
            st.success("تم تحميل config.json")
            st.rerun()
        except Exception as e:
            st.error(f"فشل تحميل config: {e}")

st.divider()
st.subheader("التصحيح وإخراج Excel")

theory_txt = st.text_input("نطاق الأسئلة التي تُحسب (مثال: 1-20 أو 1-60)", "1-20")
theory_ranges = parse_ranges(theory_txt)

if st.button("🚀 ابدأ التصحيح الآن"):
    if roster_file is None or key_file is None or sheets_file is None:
        st.error("ارفع: Roster + Answer Key + أوراق الطلاب")
        st.stop()

    if cfg.id_roi[2] <= 0 or cfg.id_roi[3] <= 0:
        st.error("حدد ID ROI (كود الطالب) أولاً.")
        st.stop()

    if not cfg.q_blocks:
        st.error("أضف بلوك/بلوكات الأسئلة أولاً.")
        st.stop()

    # Read roster
    if roster_file.name.lower().endswith(("xlsx", "xls")):
        df_roster = pd.read_excel(roster_file)
    else:
        df_roster = pd.read_csv(roster_file)

    df_roster.columns = [c.strip().lower() for c in df_roster.columns]
    if "student_code" not in df_roster.columns or "student_name" not in df_roster.columns:
        st.error("Roster يجب أن يحتوي أعمدة: student_code, student_name")
        st.stop()

    roster = dict(zip(df_roster["student_code"].astype(str), df_roster["student_name"].astype(str)))

    # Read key answers
    key_pages = load_pages(key_file.getvalue(), key_file.name)
    key_thr = preprocess(pil_to_cv(key_pages[0]))
    key_ans = read_answers(key_thr, cfg.q_blocks, choices, min_fill=min_fill_q, min_ratio=min_ratio)

    # Read student sheets
    pages = load_pages(sheets_file.getvalue(), sheets_file.name)

    results = []
    prog = st.progress(0)
    total_pages = len(pages)

    for i, pg in enumerate(pages, 1):
        thr = preprocess(pil_to_cv(pg))

        code = read_student_code(thr, cfg, min_fill=min_fill_id, min_ratio=min_ratio)
        name = roster.get(code, "")

        stu_ans = read_answers(thr, cfg.q_blocks, choices, min_fill=min_fill_q, min_ratio=min_ratio)

        total_q = 0
        score = 0
        for q, (ka, _) in key_ans.items():
            if theory_ranges and not in_ranges(q, theory_ranges):
                continue
            total_q += 1
            sa, stt = stu_ans.get(q, ("?", "MISSING"))
            if sa == ka:
                score += 1

        results.append({
            "sheet_index": i,
            "student_code": code,
            "student_name": name,
            "score": score,
            "total_questions": total_q
        })

        prog.progress(int(i / total_pages * 100))

    out = pd.DataFrame(results)
    st.success("✅ تم التصحيح")
    st.dataframe(out, use_container_width=True)

    buf = io.BytesIO()
    out.to_excel(buf, index=False)
    st.download_button("تحميل Excel", buf.getvalue(), "results.xlsx")
