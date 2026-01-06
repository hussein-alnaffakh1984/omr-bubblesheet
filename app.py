# app.py  (FULL REPLACEMENT - Calibrated OMR)
# ------------------------------------------
# ✅ Reads student code
# ✅ Reads answers + grades vs answer key
# ✅ Exports Excel: sheet_index, student_code, student_name, score
# ✅ Built-in calibration sliders for ROI (fixes wrong codes/scores)
# ✅ Debug preview for first page + key page
# ✅ Batch pages to avoid Streamlit Cloud memory issues

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
# Reference Sheet Size (Your sample)
# =========================
REF_W, REF_H = 991, 1420  # keep ثابت


# =========================
# Template Config
# =========================
@dataclass
class TemplateConfig:
    id_roi: Tuple[int, int, int, int]                 # x,y,w,h
    id_digits: int                                   # columns
    id_rows: int                                     # 0-9 rows

    q_blocks: List[Tuple[int, int, int, int, int, int]]  # x,y,w,h, qs,qe
    block_rows: int                                  # questions per block (vertical)


def default_cfg() -> TemplateConfig:
    # ✅ IMPORTANT:
    # Updated ID ROI to cover FULL 4 columns (more correct than old 320 width)
    # based on your page1 sample.
    return TemplateConfig(
        id_roi=(540, 120, 430, 450),   # ✅ improved
        id_digits=4,
        id_rows=10,
        q_blocks=[
            (120, 520, 260, 860, 1, 20),
            (440, 520, 260, 860, 21, 40),
            (760, 520, 220, 860, 41, 60),
        ],
        block_rows=20
    )


BASE = default_cfg()


def scale_box(box, sx, sy):
    x, y, w, h = box
    return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))


def cfg_for_image(img_w, img_h, id_offset=(0, 0, 0, 0), b_offset=(0, 0, 0, 0)):
    """
    Scale reference ROIs to current image size + apply manual offsets from sliders.
    Offsets are applied in REF coordinate space first then scaled.
    """
    sx, sy = img_w / REF_W, img_h / REF_H

    # apply offsets on REF coords
    ix, iy, iw, ih = BASE.id_roi
    ox, oy, ow, oh = id_offset
    id_ref = (ix + ox, iy + oy, iw + ow, ih + oh)
    id_scaled = scale_box(id_ref, sx, sy)

    blocks_scaled = []
    bx, by, bw, bh = b_offset
    for (x, y, w, h, qs, qe) in BASE.q_blocks:
        block_ref = (x + bx, y + by, w + bw, h + bh)
        xs, ys, ws, hs = scale_box(block_ref, sx, sy)
        blocks_scaled.append((xs, ys, ws, hs, qs, qe))

    return TemplateConfig(
        id_roi=id_scaled,
        id_digits=BASE.id_digits,
        id_rows=BASE.id_rows,
        q_blocks=blocks_scaled,
        block_rows=BASE.block_rows
    )


# =========================
# Image utils
# =========================
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """
    Binary inverted: marks become white pixels (255)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )
    return thr


def score_cell(bin_img: np.ndarray) -> int:
    # count white pixels
    return int((bin_img > 0).sum())


def pick_one(scores: List[Tuple[str, int]], min_fill: int, min_ratio: float):
    """
    Returns (char, status)
    status: OK / BLANK / DOUBLE
    """
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0

    if top_s < min_fill:
        return "?", "BLANK"
    if second_s > 0 and (top_s / (second_s + 1e-6)) < min_ratio:
        return "!", "DOUBLE"
    return top_c, "OK"


# =========================
# Readers
# =========================
def read_student_code(thr: np.ndarray, cfg: TemplateConfig,
                      min_fill: int, min_ratio: float):
    """
    Reads 4-digit code from 10x4 bubble grid.
    """
    x, y, w, h = cfg.id_roi
    roi = thr[y:y + h, x:x + w]

    rows, cols = cfg.id_rows, cfg.id_digits
    ch, cw = h // rows, w // cols

    digits = []
    statuses = []
    for c in range(cols):
        col_scores = []
        for r in range(rows):
            cell = roi[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            col_scores.append((str(r), score_cell(cell)))

        d, stt = pick_one(col_scores, min_fill=min_fill, min_ratio=min_ratio)
        digits.append("" if d in ["?", "!"] else d)
        statuses.append(stt)

    code = "".join(digits).strip()

    # IMPORTANT: do NOT force zfill if code invalid
    # If any digit blank, consider code invalid
    if len(code) != cfg.id_digits:
        return "", statuses

    return code, statuses


def read_answers(thr: np.ndarray, cfg: TemplateConfig, choices: int,
                 min_fill: int, min_ratio: float):
    letters = "ABCDE"[:choices]
    out: Dict[int, Tuple[str, str]] = {}

    for (x, y, w, h, qs, qe) in cfg.q_blocks:
        roi = thr[y:y + h, x:x + w]
        rows = cfg.block_rows
        rh, cw = h // rows, w // choices

        q = qs
        for r in range(rows):
            if q > qe:
                break

            scores = []
            for c in range(choices):
                cell = roi[r * rh:(r + 1) * rh, c * cw:(c + 1) * cw]
                scores.append((letters[c], score_cell(cell)))

            a, stt = pick_one(scores, min_fill=min_fill, min_ratio=min_ratio)
            out[q] = (a, stt)
            q += 1

    return out


def parse_ranges(txt: str):
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


def in_ranges(q: int, ranges):
    return any(a <= q <= b for a, b in ranges)


def compute_score(key_ans, stu_ans, theory_ranges, practical_ranges):
    score = 0
    grade_all = (not theory_ranges and not practical_ranges)

    for q, (ka, _) in key_ans.items():
        sa, stt = stu_ans.get(q, ("?", "BLANK"))

        if sa in ["?", "!"]:
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
def load_pages(file_bytes: bytes, filename: str, dpi: int, first_page: int = 1, last_page: int = 1):
    if filename.lower().endswith(".pdf"):
        return convert_from_bytes(file_bytes, dpi=dpi, first_page=first_page, last_page=last_page)
    return [Image.open(io.BytesIO(file_bytes))]


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="OMR Bubble Sheet", layout="wide")
st.title("تصحيح ببل شيت (مع معايرة داخل البرنامج)")

left, right = st.columns([1, 1])

with left:
    st.subheader("⚙️ الإعدادات")

    choices = st.radio("عدد الخيارات", [4, 5], horizontal=True)
    dpi = st.slider("DPI تحويل PDF (أقل أسرع وأخف)", 80, 200, 120, 10)

    theory_txt = st.text_input("نطاق النظري (مثال: 1-60) — إذا فارغ يصحح الكل", "")
    practical_txt = st.text_input("نطاق العملي (اختياري)", "")

    theory_ranges = parse_ranges(theory_txt)
    practical_ranges = parse_ranges(practical_txt)

    st.markdown("---")
    st.subheader("🎛 حساسية التظليل")
    # These are critical for accuracy:
    id_min_fill = st.slider("ID min_fill", 50, 2000, 350, 10)
    id_min_ratio = st.slider("ID min_ratio", 1.05, 2.50, 1.25, 0.05)

    ans_min_fill = st.slider("Answers min_fill", 50, 2000, 220, 10)
    ans_min_ratio = st.slider("Answers min_ratio", 1.05, 2.50, 1.25, 0.05)

    st.markdown("---")
    st.subheader("🧩 معايرة المكان (ROI)")
    st.caption("حرّك السلايدرات حتى تشوف الشبكة/الفقاعات داخل القص بشكل صحيح")

    id_dx = st.slider("ID x offset", -200, 200, 0, 5)
    id_dy = st.slider("ID y offset", -200, 200, 0, 5)
    id_dw = st.slider("ID w offset", -200, 200, 0, 5)
    id_dh = st.slider("ID h offset", -200, 200, 0, 5)

    b_dx = st.slider("Blocks x offset", -200, 200, 0, 5)
    b_dy = st.slider("Blocks y offset", -200, 200, 0, 5)
    b_dw = st.slider("Blocks w offset", -200, 200, 0, 5)
    b_dh = st.slider("Blocks h offset", -200, 200, 0, 5)

    debug = st.checkbox("Debug: عرض القص + ما تم قراءته", True)

    st.markdown("---")
    st.subheader("📄 Batch")
    start_page = st.number_input("Start page", min_value=1, value=1, step=1)
    end_page = st.number_input("End page", min_value=1, value=50, step=1)

with right:
    st.subheader("📥 الملفات")

    roster_file = st.file_uploader("1) Roster (Excel/CSV) يحتوي: student_code, student_name",
                                   type=["xlsx", "xls", "csv"])

    key_file = st.file_uploader("2) Answer Key (PDF صفحة واحدة أو صورة)",
                                type=["pdf", "png", "jpg", "jpeg"])

    sheets_file = st.file_uploader("3) Student Sheets (PDF متعدد الصفحات أو صور)",
                                   type=["pdf", "png", "jpg", "jpeg"])

    st.markdown("---")

# Load roster
roster = {}
if roster_file:
    fn = roster_file.name.lower()
    df = pd.read_csv(roster_file) if fn.endswith(".csv") else pd.read_excel(roster_file)
    df.columns = [c.strip().lower() for c in df.columns]

    if "student_code" not in df.columns or "student_name" not in df.columns:
        st.error("Roster لازم يحتوي عمودين: student_code و student_name")
        st.stop()

    df["student_code"] = df["student_code"].astype(str).str.strip()
    df["student_name"] = df["student_name"].astype(str).str.strip()

    # do not force zfill here unless your codes فعلاً 4 digits
    # if you want fixed length uncomment:
    # df["student_code"] = df["student_code"].apply(lambda x: x.zfill(BASE.id_digits))

    roster = dict(zip(df["student_code"], df["student_name"]))
    st.success(f"✅ Loaded roster: {len(roster)} students")


def crop_preview(thr, box):
    x, y, w, h = box
    h0, w0 = thr.shape[:2]
    x = max(0, min(x, w0 - 1))
    y = max(0, min(y, h0 - 1))
    w = max(1, min(w, w0 - x))
    h = max(1, min(h, h0 - y))
    return thr[y:y + h, x:x + w]


if st.button("🚀 ابدأ التصحيح"):
    if not roster_file or not key_file or not sheets_file:
        st.error("ارفع Roster + Answer Key + Student Sheets")
        st.stop()

    if end_page < start_page:
        st.error("End page لازم >= Start page")
        st.stop()

    # 1) Key
    key_pages = load_pages(key_file.getvalue(), key_file.name, dpi=dpi, first_page=1, last_page=1)
    key_img = key_pages[0]
    key_thr = preprocess(pil_to_bgr(key_img))

    Hk, Wk = key_thr.shape[:2]
    cfgk = cfg_for_image(
        Wk, Hk,
        id_offset=(id_dx, id_dy, id_dw, id_dh),
        b_offset=(b_dx, b_dy, b_dw, b_dh)
    )

    key_ans = read_answers(key_thr, cfgk, choices=choices, min_fill=ans_min_fill, min_ratio=ans_min_ratio)

    if debug:
        st.markdown("### 🔎 Debug (Answer Key)")
        st.image(crop_preview(key_thr, cfgk.id_roi), caption="Key - ID ROI (not used)", clamp=True)
        for i, (x, y, w, h, qs, qe) in enumerate(cfgk.q_blocks, 1):
            st.image(key_thr[y:y + h, x:x + w], caption=f"Key - Block {i} ({qs}-{qe})", clamp=True)
        # Show first 10 answers
        st.write("Key (first 10):", {k: key_ans[k] for k in sorted(key_ans)[:10]})

    # 2) Student pages (batch)
    if sheets_file.name.lower().endswith(".pdf"):
        pages = load_pages(
            sheets_file.getvalue(),
            sheets_file.name,
            dpi=dpi,
            first_page=int(start_page),
            last_page=int(end_page)
        )
    else:
        pages = [Image.open(io.BytesIO(sheets_file.getvalue()))]

    results = []
    prog = st.progress(0)
    total = len(pages)

    for idx, pg in enumerate(pages, 1):
        sheet_index = int(start_page) + idx - 1

        thr = preprocess(pil_to_bgr(pg))
        H, W = thr.shape[:2]
        cfg = cfg_for_image(
            W, H,
            id_offset=(id_dx, id_dy, id_dw, id_dh),
            b_offset=(b_dx, b_dy, b_dw, b_dh)
        )

        code, code_statuses = read_student_code(thr, cfg, min_fill=id_min_fill, min_ratio=id_min_ratio)
        name = roster.get(code, "")

        stu_ans = read_answers(thr, cfg, choices=choices, min_fill=ans_min_fill, min_ratio=ans_min_ratio)
        score = compute_score(key_ans, stu_ans, theory_ranges, practical_ranges)

        results.append({
            "sheet_index": sheet_index,
            "student_code": code,
            "student_name": name,
            "score": score
        })

        if debug and idx == 1:
            st.markdown("### 🔎 Debug (First Student Page)")
            st.image(crop_preview(thr, cfg.id_roi), caption=f"Student - ID ROI | read={code} | statuses={code_statuses}", clamp=True)
            for i, (x, y, w, h, qs, qe) in enumerate(cfg.q_blocks, 1):
                st.image(thr[y:y + h, x:x + w], caption=f"Student - Block {i} ({qs}-{qe})", clamp=True)
            # Show first 10 answers
            st.write("Student (first 10):", {k: stu_ans[k] for k in sorted(stu_ans)[:10]})

        prog.progress(int(idx / total * 100))

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
