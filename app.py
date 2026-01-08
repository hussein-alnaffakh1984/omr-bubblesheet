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

# Optional but recommended (you already added it)
from streamlit_image_coordinates import streamlit_image_coordinates


# =========================
# Data Structures
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
    template_w: int = 0
    template_h: int = 0

    # Student ID region
    id_roi: Tuple[int, int, int, int] = (0, 0, 0, 0)
    id_digits: int = 4
    id_rows: int = 10  # 0..9

    # Question blocks
    q_blocks: List[QBlock] = None

    # bubble choices in each question row
    choices: int = 4

    def to_jsonable(self):
        d = asdict(self)
        d["q_blocks"] = [asdict(b) for b in (self.q_blocks or [])]
        return d

    @staticmethod
    def from_jsonable(d: dict):
        cfg = TemplateConfig()
        cfg.template_w = int(d.get("template_w", 0))
        cfg.template_h = int(d.get("template_h", 0))
        cfg.id_roi = tuple(d.get("id_roi", (0, 0, 0, 0)))
        cfg.id_digits = int(d.get("id_digits", 4))
        cfg.id_rows = int(d.get("id_rows", 10))
        cfg.choices = int(d.get("choices", 4))
        cfg.q_blocks = [QBlock(**b) for b in d.get("q_blocks", [])]
        return cfg


# =========================
# Helpers: Images / PDF
# =========================
def load_pages(file_bytes: bytes, filename: str) -> List[Image.Image]:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, fmt="png")
        return pages
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def resize_to(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)


# =========================
# Alignment (VERY IMPORTANT)
# =========================
def order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def find_page_quad(bgr: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)

    # close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * (bgr.shape[0] * bgr.shape[1]):
            pts = approx.reshape(4, 2).astype(np.float32)
            return order_points(pts)

    return None

def warp_to_template(bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    quad = find_page_quad(bgr)
    if quad is None:
        # fallback: resize only
        return resize_to(bgr, tw, th)

    dst = np.array([
        [0, 0],
        [tw - 1, 0],
        [tw - 1, th - 1],
        [0, th - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(bgr, M, (tw, th))
    return warped


# =========================
# Preprocess & Bubble Scoring
# =========================
def preprocess_for_bubbles(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold (invert => filled becomes white)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )
    # remove tiny noise
    thr = cv2.medianBlur(thr, 3)
    return thr

def inner_crop(cell: np.ndarray, margin_ratio: float = 0.22) -> np.ndarray:
    h, w = cell.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    return cell[my:h - my, mx:w - mx]

def score_cell(bin_cell: np.ndarray) -> float:
    # bin_cell: white pixels represent ink after THRESH_BINARY_INV
    c = inner_crop(bin_cell, 0.22)
    return float(np.sum(c > 0)) / (c.shape[0] * c.shape[1] + 1e-9)

def pick_one(scores: List[Tuple[str, float]], min_fill=0.20, min_ratio=1.35):
    # scores: [(choice, fill_ratio)]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0.0

    if top_s < min_fill:
        return "?", "BLANK", top_s, second_s
    if second_s > 0 and (top_s / (second_s + 1e-9)) < min_ratio:
        return "!", "DOUBLE", top_s, second_s
    return top_c, "OK", top_s, second_s


# =========================
# Read Student Code
# =========================
def read_student_code(thr: np.ndarray, cfg: TemplateConfig) -> Tuple[str, Dict]:
    x, y, w, h = cfg.id_roi
    if w <= 0 or h <= 0:
        return "", {"error": "ID ROI not set"}

    roi = thr[y:y + h, x:x + w]
    rows = cfg.id_rows
    cols = cfg.id_digits
    ch = h // rows
    cw = w // cols

    digits = []
    debug_cols = []

    for c in range(cols):
        col_scores = []
        for r in range(rows):
            cell = roi[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            fill = score_cell(cell)
            col_scores.append((str(r), fill))
        d, status, top, second = pick_one(col_scores, min_fill=0.18, min_ratio=1.25)
        digits.append("" if d in ("?", "!") else d)
        debug_cols.append({"col": c, "status": status, "top": top, "second": second, "scores": col_scores})

    code = "".join(digits)
    return code, {"cols": debug_cols, "raw": digits}


# =========================
# Read Answers
# =========================
def read_answers(thr: np.ndarray, block: QBlock, choices: int) -> Dict[int, Tuple[str, str]]:
    letters = "ABCDE"[:choices]
    out = {}

    x, y, w, h = block.x, block.y, block.w, block.h
    roi = thr[y:y + h, x:x + w]

    rows = block.rows
    rh = h // rows
    cw = w // choices

    q = block.start_q
    for r in range(rows):
        if q > block.end_q:
            break
        scores = []
        for c in range(choices):
            cell = roi[r * rh:(r + 1) * rh, c * cw:(c + 1) * cw]
            scores.append((letters[c], score_cell(cell)))
        a, status, _, _ = pick_one(scores, min_fill=0.20, min_ratio=1.35)
        out[q] = (a, status)
        q += 1
    return out


# =========================
# Ranges
# =========================
def parse_ranges(txt: str) -> List[Tuple[int, int]]:
    if not (txt or "").strip():
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
        return False
    return any(a <= q <= b for a, b in ranges)


# =========================
# Draw Preview
# =========================
def draw_cfg_preview(img: Image.Image, cfg: TemplateConfig) -> Image.Image:
    im = img.copy().convert("RGB")
    dr = ImageDraw.Draw(im)

    # ID ROI in red
    x, y, w, h = cfg.id_roi
    if w > 0 and h > 0:
        dr.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=4)
        dr.text((x + 5, y + 5), "ID ROI", fill=(255, 0, 0))

    # Q blocks in green
    for i, b in enumerate(cfg.q_blocks or [], 1):
        dr.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=(0, 180, 0), width=4)
        dr.text((b.x + 5, b.y + 5), f"Q{i}: {b.start_q}-{b.end_q}", fill=(0, 140, 0))
    return im


# =========================
# UI
# =========================
st.set_page_config(page_title="OMR Bubble Sheet (Remark-Style)", layout="wide")

st.markdown(
    """
    <style>
      .small-note {opacity:0.75; font-size: 0.92rem;}
      .block-title {font-weight:800; font-size:1.2rem;}
      .stButton>button {border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("✅ OMR Bubble Sheet — واجهة مثل Remark (تحديد بنقطتين + تصحيح + Excel)")

# Session state
if "cfg" not in st.session_state:
    st.session_state.cfg = TemplateConfig(q_blocks=[])

if "clicks" not in st.session_state:
    st.session_state.clicks = []  # [(x,y), (x,y)]

if "template_img" not in st.session_state:
    st.session_state.template_img = None

if "template_bytes" not in st.session_state:
    st.session_state.template_bytes = None

if "template_name" not in st.session_state:
    st.session_state.template_name = ""


left, right = st.columns([1.55, 1], gap="large")

with right:
    st.markdown('<div class="block-title">1) رفع نموذج الورقة (Template)</div>', unsafe_allow_html=True)
    tpl = st.file_uploader("PDF/PNG/JPG (صفحة النموذج)", type=["pdf", "png", "jpg", "jpeg"], key="tpl_upl")

    st.markdown('<div class="block-title" style="margin-top:18px;">2) إعدادات عامة</div>', unsafe_allow_html=True)
    canvas_w = st.slider("عرض المعاينة (700 مناسب)", 500, 1400, 700, 10)
    choices = st.radio("عدد الخيارات", [4, 5], horizontal=True, index=0)
    id_digits = st.number_input("عدد خانات كود الطالب", min_value=1, max_value=12, value=int(st.session_state.cfg.id_digits), step=1)
    id_rows = st.number_input("عدد صفوف أرقام الكود (عادة 10)", min_value=5, max_value=15, value=int(st.session_state.cfg.id_rows), step=1)

    st.session_state.cfg.choices = int(choices)
    st.session_state.cfg.id_digits = int(id_digits)
    st.session_state.cfg.id_rows = int(id_rows)

    st.markdown('<div class="block-title" style="margin-top:18px;">3) إعداد بلوك الأسئلة</div>', unsafe_allow_html=True)
    b_start = st.number_input("Start Q", min_value=1, max_value=500, value=1, step=1)
    b_end = st.number_input("End Q", min_value=1, max_value=500, value=20, step=1)
    b_rows = st.number_input("Rows داخل البلوك", min_value=1, max_value=200, value=20, step=1)

    mode = st.radio("ماذا نحدد الآن؟", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], index=0)

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("مسح آخر نقطة"):
            if st.session_state.clicks:
                st.session_state.clicks.pop()
    with colB:
        if st.button("Reset الكل"):
            st.session_state.clicks = []
            st.session_state.cfg.id_roi = (0, 0, 0, 0)
            st.session_state.cfg.q_blocks = []
    with colC:
        st.write("")

    st.markdown('<div class="small-note">طريقة التحديد: اضغط نقطتين على الصورة (زاوية 1 ثم زاوية 2) لتكوين مستطيل.</div>', unsafe_allow_html=True)

    if st.button("💾 حفظ المستطيل الحالي"):
        if len(st.session_state.clicks) < 2:
            st.error("لازم تختار نقطتين أولاً.")
        else:
            (x1, y1), (x2, y2) = st.session_state.clicks[-2], st.session_state.clicks[-1]
            x = int(min(x1, x2))
            y = int(min(y1, y2))
            w = int(abs(x2 - x1))
            h = int(abs(y2 - y1))

            if w < 5 or h < 5:
                st.error("المستطيل صغير جدًا.")
            else:
                if mode.startswith("ID"):
                    st.session_state.cfg.id_roi = (x, y, w, h)
                    st.success("تم حفظ ID ROI ✅")
                else:
                    qb = QBlock(
                        x=x, y=y, w=w, h=h,
                        start_q=int(min(b_start, b_end)),
                        end_q=int(max(b_start, b_end)),
                        rows=int(b_rows)
                    )
                    st.session_state.cfg.q_blocks.append(qb)
                    st.success("تم إضافة Q Block ✅")

    if st.button("🗑️ حذف آخر Q Block"):
        if st.session_state.cfg.q_blocks:
            st.session_state.cfg.q_blocks.pop()
            st.success("تم حذف آخر بلوك.")
        else:
            st.info("لا يوجد بلوكات.")

    st.markdown('<div class="block-title" style="margin-top:18px;">4) ملفات التصحيح</div>', unsafe_allow_html=True)
    roster_file = st.file_uploader("Roster Excel/CSV: student_code, student_name", type=["xlsx", "xls", "csv"], key="roster_upl")
    key_file = st.file_uploader("Answer Key (نفس النموذج) PDF/صورة", type=["pdf", "png", "jpg", "jpeg"], key="key_upl")
    sheets_file = st.file_uploader("أوراق الطلاب PDF/صور", type=["pdf", "png", "jpg", "jpeg"], key="sheets_upl")

    st.markdown('<div class="block-title" style="margin-top:18px;">5) نطاقات الدرجات</div>', unsafe_allow_html=True)
    theory_txt = st.text_input("نطاق النظري (مثال: 1-40)", "")
    practical_txt = st.text_input("نطاق العملي (اختياري)", "")
    strict = st.checkbox("وضع صارم (BLANK/DOUBLE = خطأ)", True)

with left:
    if tpl is not None:
        st.session_state.template_bytes = tpl.getvalue()
        st.session_state.template_name = tpl.name
        pages = load_pages(st.session_state.template_bytes, st.session_state.template_name)
        st.session_state.template_img = pages[0].convert("RGB")
        tw, th = st.session_state.template_img.size
        st.session_state.cfg.template_w = tw
        st.session_state.cfg.template_h = th

    if st.session_state.template_img is None:
        st.info("ارفع Template أولاً من اليمين.")
        st.stop()

    # Preview with ROIs
    preview = draw_cfg_preview(st.session_state.template_img, st.session_state.cfg)

    # Click-to-get-coordinates
    st.markdown('<div class="block-title">واجهة التحديد (اضغط نقطتين)</div>', unsafe_allow_html=True)
    coords = streamlit_image_coordinates(preview, width=canvas_w, key="img_coords")

    if coords is not None and "x" in coords and "y" in coords:
        # coords are in rendered size; convert to original size
        orig_w, orig_h = st.session_state.template_img.size
        scale = orig_w / float(canvas_w)
        x = int(coords["x"] * scale)
        y = int(coords["y"] * scale)
        st.session_state.clicks.append((x, y))
        st.caption(f"📍 Click: ({x}, {y}) — total clicks: {len(st.session_state.clicks)}")

    # Show current rectangle (last 2 clicks)
    if len(st.session_state.clicks) >= 2:
        (x1, y1), (x2, y2) = st.session_state.clicks[-2], st.session_state.clicks[-1]
        st.caption(f"🧩 المستطيل الحالي: x={min(x1,x2)}, y={min(y1,y2)}, w={abs(x2-x1)}, h={abs(y2-y1)}")

    st.markdown("---")

    # =========================
    # Grading
    # =========================
    if st.button("🚀 ابدأ التصحيح"):
        cfg = st.session_state.cfg

        # Validate config
        if cfg.template_w <= 0 or cfg.template_h <= 0:
            st.error("Template غير صالح.")
            st.stop()

        if cfg.id_roi[2] <= 0 or cfg.id_roi[3] <= 0:
            st.error("حدد ID ROI أولاً.")
            st.stop()

        if not cfg.q_blocks:
            st.error("أضف على الأقل Q Block واحد.")
            st.stop()

        if roster_file is None or key_file is None or sheets_file is None:
            st.error("ارفع Roster + Answer Key + أوراق الطلاب.")
            st.stop()

        # Load roster
        if roster_file.name.lower().endswith(("xlsx", "xls")):
            df_roster = pd.read_excel(roster_file)
        else:
            df_roster = pd.read_csv(roster_file)

        df_roster.columns = [c.strip().lower() for c in df_roster.columns]
        if "student_code" not in df_roster.columns or "student_name" not in df_roster.columns:
            st.error("Roster لازم يحتوي عمودين: student_code و student_name")
            st.stop()

        roster = dict(
            zip(df_roster["student_code"].astype(str).str.strip(), df_roster["student_name"].astype(str).str.strip())
        )

        # Load key page
        key_pages = load_pages(key_file.getvalue(), key_file.name)
        key_bgr = pil_to_bgr(key_pages[0])
        key_bgr = warp_to_template(key_bgr, cfg.template_w, cfg.template_h)
        key_thr = preprocess_for_bubbles(key_bgr)

        # Read key answers
        key_ans = {}
        for b in cfg.q_blocks:
            key_ans.update(read_answers(key_thr, b, cfg.choices))

        # Prepare ranges
        theory_ranges = parse_ranges(theory_txt)
        practical_ranges = parse_ranges(practical_txt)

        # Load student pages
        pages = load_pages(sheets_file.getvalue(), sheets_file.name)

        prog = st.progress(0)
        results = []
        total_pages = len(pages)

        for idx, pg in enumerate(pages, 1):
            bgr = pil_to_bgr(pg)
            bgr = warp_to_template(bgr, cfg.template_w, cfg.template_h)
            thr = preprocess_for_bubbles(bgr)

            code, code_dbg = read_student_code(thr, cfg)
            code = (code or "").strip()
            if code == "":
                code = "UNKNOWN"

            name = roster.get(code, "")

            # Read student answers
            stu_ans = {}
            for b in cfg.q_blocks:
                stu_ans.update(read_answers(thr, b, cfg.choices))

            # Score
            score = 0
            total_q = 0

            for q, (ka, _) in key_ans.items():
                # if user provided ranges -> only count those
                use_q = False
                if theory_ranges and in_ranges(q, theory_ranges):
                    use_q = True
                if practical_ranges and in_ranges(q, practical_ranges):
                    use_q = True

                # if no ranges at all -> count all
                if not theory_ranges and not practical_ranges:
                    use_q = True

                if not use_q:
                    continue

                total_q += 1
                sa, stt = stu_ans.get(q, ("?", "MISSING"))

                if strict and stt in ("BLANK", "DOUBLE"):
                    continue

                if sa == ka:
                    score += 1

            results.append({
                "sheet_index": idx,
                "student_code": code,
                "student_name": name,
                "score": score,
                "total_questions": total_q
            })

            prog.progress(int(idx / total_pages * 100))

        out = pd.DataFrame(results)

        # Export Excel
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="Results")

        st.success("✅ تم إنشاء ملف Excel")
        st.download_button("⬇️ تحميل النتائج Excel", buf.getvalue(), "results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
