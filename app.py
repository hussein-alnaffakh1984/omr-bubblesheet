import io
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from streamlit_drawable_canvas import st_canvas


# =========================
# Helpers
# =========================
def load_pages(file_bytes: bytes, filename: str, dpi: int,
               first_page: int = 1, last_page: int = 1) -> List[Image.Image]:
    fn = filename.lower()
    if fn.endswith(".pdf"):
        return convert_from_bytes(file_bytes, dpi=dpi, first_page=first_page, last_page=last_page)
    return [Image.open(io.BytesIO(file_bytes))]


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )
    return thr


def cell_fill_ratio(bin_img: np.ndarray) -> float:
    return float((bin_img > 0).mean())


def pick_one_ratio(scores: List[Tuple[str, float]], blank_thr: float, min_ratio: float):
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0.0

    if top_s < blank_thr:
        return "?", "BLANK"

    if second_s > 0 and (top_s / (second_s + 1e-9)) < min_ratio:
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


# =========================
# Template Data
# =========================
@dataclass
class QBlock:
    name: str
    roi: Tuple[int, int, int, int]  # x,y,w,h
    q_start: int
    q_end: int
    rows: int


@dataclass
class Template:
    base_w: int
    base_h: int
    id_roi: Tuple[int, int, int, int]
    id_digits: int
    id_rows: int
    q_blocks: List[QBlock]
    choices_default: int = 4


def scale_roi(roi, sx, sy):
    x, y, w, h = roi
    return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))


def template_for_page(tpl: Template, page_w: int, page_h: int) -> Template:
    sx, sy = page_w / tpl.base_w, page_h / tpl.base_h
    id_roi = scale_roi(tpl.id_roi, sx, sy)
    q_blocks = []
    for b in tpl.q_blocks:
        q_blocks.append(QBlock(
            name=b.name,
            roi=scale_roi(b.roi, sx, sy),
            q_start=b.q_start,
            q_end=b.q_end,
            rows=b.rows
        ))
    return Template(
        base_w=page_w,
        base_h=page_h,
        id_roi=id_roi,
        id_digits=tpl.id_digits,
        id_rows=tpl.id_rows,
        q_blocks=q_blocks,
        choices_default=tpl.choices_default
    )


# =========================
# OMR Readers
# =========================
def read_student_code(thr: np.ndarray, tpl: Template,
                      id_blank_thr: float, id_min_ratio: float) -> Tuple[str, List[str]]:
    x, y, w, h = tpl.id_roi
    roi = thr[y:y + h, x:x + w]

    rows, cols = tpl.id_rows, tpl.id_digits
    ch, cw = max(1, h // rows), max(1, w // cols)

    digits = []
    statuses = []

    for c in range(cols):
        col_scores = []
        for r in range(rows):
            cell = roi[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            col_scores.append((str(r), cell_fill_ratio(cell)))

        d, stt = pick_one_ratio(col_scores, blank_thr=id_blank_thr, min_ratio=id_min_ratio)
        digits.append(d if d not in ["?","!"] else "?")
        statuses.append(stt)

    return "".join(digits), statuses


def read_answers(thr: np.ndarray, tpl: Template, choices: int,
                 ans_blank_thr: float, ans_min_ratio: float) -> Dict[int, Tuple[str, str]]:
    letters = "ABCDE"[:choices]
    out: Dict[int, Tuple[str, str]] = {}

    for block in tpl.q_blocks:
        x, y, w, h = block.roi
        roi = thr[y:y + h, x:x + w]

        rows = block.rows
        rh = max(1, h // rows)
        cw = max(1, w // choices)

        q = block.q_start
        for r in range(rows):
            if q > block.q_end:
                break
            scores = []
            for c in range(choices):
                cell = roi[r * rh:(r + 1) * rh, c * cw:(c + 1) * cw]
                scores.append((letters[c], cell_fill_ratio(cell)))
            a, stt = pick_one_ratio(scores, blank_thr=ans_blank_thr, min_ratio=ans_min_ratio)
            out[q] = (a, stt)
            q += 1

    return out


def compute_score(key_ans: Dict[int, Tuple[str, str]],
                  stu_ans: Dict[int, Tuple[str, str]],
                  theory_ranges: List[Tuple[int, int]],
                  practical_ranges: List[Tuple[int, int]]) -> int:
    grade_all = (not theory_ranges and not practical_ranges)
    score = 0

    for q, (ka, _) in key_ans.items():
        sa, _ = stu_ans.get(q, ("?", "BLANK"))
        if sa in ["?","!"]:
            continue

        if grade_all:
            score += int(sa == ka)
            continue

        if theory_ranges and in_ranges(q, theory_ranges):
            score += int(sa == ka)
        if practical_ranges and in_ranges(q, practical_ranges):
            score += int(sa == ka)

    return score


# =========================
# Canvas helpers
# =========================
def rects_from_canvas(canvas_json) -> List[Tuple[int, int, int, int]]:
    if not canvas_json:
        return []
    objs = canvas_json.get("objects", [])
    rects = []
    for o in objs:
        if o.get("type") != "rect":
            continue
        left = int(o.get("left", 0))
        top = int(o.get("top", 0))
        w = int(o.get("width", 0) * o.get("scaleX", 1))
        h = int(o.get("height", 0) * o.get("scaleY", 1))
        rects.append((left, top, w, h))
    return rects


def cloud_safe_image(img: Image.Image) -> Image.Image:
    # Fix for Streamlit Cloud canvas background bug:
    # Convert PIL image -> PNG bytes -> reload PIL
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf)


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Remark-Style OMR", layout="wide")
st.title("OMR Bubble Sheet — Remark-Style (رسم يدوي + تصحيح + Excel)")

st.caption("✅ ارسم مناطق الكود والاسئلة بالماوس مثل Remark ثم صحّح وصدّر Excel. "
           "ملاحظة: Streamlit Cloud يقيّد رفع الملفات الكبيرة 200MB للملف الواحد.")

tab1, tab2 = st.tabs(["① بناء Template", "② التصحيح وExcel"])

# =========================
# TAB 1: Build Template
# =========================
with tab1:
    st.subheader("1) ارفع صفحة نموذج واحدة (Template)")
    dpi_t = st.slider("DPI للعرض (Template)", 80, 200, 120, 10, key="dpi_t")

    tpl_file = st.file_uploader("Template page (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"], key="tpl_file")

    if "tpl_id_roi" not in st.session_state:
        st.session_state.tpl_id_roi = None
    if "tpl_blocks" not in st.session_state:
        st.session_state.tpl_blocks = []

    if tpl_file:
        pages = load_pages(tpl_file.getvalue(), tpl_file.name, dpi=dpi_t, first_page=1, last_page=1)
        tpl_img = pages[0]
        base_w, base_h = tpl_img.size

        st.write(f"Template image size: {base_w} x {base_h}")

        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            mode = st.radio("ماذا ترسم الآن؟", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], horizontal=False)
        with colB:
            id_digits = st.number_input("عدد خانات الكود", 3, 12, 4, 1)
            id_rows = st.number_input("عدد صفوف أرقام الكود (عادة 10)", 8, 12, 10, 1)
        with colC:
            choices_default = st.radio("عدد الخيارات الافتراضي", [4, 5], horizontal=True)

        st.markdown("### ارسم مستطيلات على الصورة (اسحب بالماوس)")
        safe_img = cloud_safe_image(tpl_img)

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.08)",
            stroke_width=2,
            stroke_color="red",
            background_image=safe_img,   # ✅ Cloud-safe
            update_streamlit=True,
            height=safe_img.size[1],
            width=safe_img.size[0],
            drawing_mode="rect",
            key="canvas_tpl",
        )

        rects = rects_from_canvas(canvas_result.json_data)

        st.markdown("---")
        st.subheader("2) تحويل آخر مستطيل إلى عنصر")

        if rects:
            latest = rects[-1]
            st.write("آخر مستطيل:", latest)

            if mode == "ID ROI (كود الطالب)":
                if st.button("✅ اجعل آخر مستطيل = منطقة كود الطالب"):
                    st.session_state.tpl_id_roi = latest
                    st.success(f"تم تعيين ID ROI = {latest}")
            else:
                c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
                with c1:
                    block_name = st.text_input("اسم البلوك", value=f"Block{len(st.session_state.tpl_blocks) + 1}")
                with c2:
                    q_start = st.number_input("من سؤال", min_value=1, value=1, step=1, key="q_start_tmp")
                with c3:
                    q_end = st.number_input("إلى سؤال", min_value=1, value=20, step=1, key="q_end_tmp")
                with c4:
                    rows = st.number_input("عدد الصفوف داخل البلوك", min_value=1,
                                           value=max(1, int(q_end) - int(q_start) + 1), step=1)

                if st.button("➕ أضف آخر مستطيل كـ Q Block"):
                    st.session_state.tpl_blocks.append({
                        "name": block_name,
                        "roi": latest,
                        "q_start": int(q_start),
                        "q_end": int(q_end),
                        "rows": int(rows)
                    })
                    st.success(f"تمت إضافة بلوك: {block_name} ROI={latest} Q={q_start}-{q_end} rows={rows}")

        st.markdown("---")
        st.subheader("3) Template الحالي")
        st.write("ID ROI:", st.session_state.tpl_id_roi)
        st.write("Q Blocks:", st.session_state.tpl_blocks)

        cX, cY = st.columns([1, 1])
        with cX:
            if st.button("🧹 مسح جميع البلوكات"):
                st.session_state.tpl_blocks = []
                st.success("تم مسح البلوكات")
        with cY:
            if st.button("🧹 مسح ID ROI"):
                st.session_state.tpl_id_roi = None
                st.success("تم مسح ID ROI")

        st.markdown("---")
        st.subheader("4) تحميل template.json")
        can_save = (st.session_state.tpl_id_roi is not None) and (len(st.session_state.tpl_blocks) > 0)

        if not can_save:
            st.warning("لازم تحدد ID ROI + على الأقل بلوك أسئلة واحد.")
        else:
            tpl_dict = {
                "base_w": int(base_w),
                "base_h": int(base_h),
                "id_roi": list(map(int, st.session_state.tpl_id_roi)),
                "id_digits": int(id_digits),
                "id_rows": int(id_rows),
                "choices_default": int(choices_default),
                "q_blocks": [
                    {
                        "name": b["name"],
                        "roi": list(map(int, b["roi"])),
                        "q_start": int(b["q_start"]),
                        "q_end": int(b["q_end"]),
                        "rows": int(b["rows"])
                    } for b in st.session_state.tpl_blocks
                ]
            }
            tpl_bytes = json.dumps(tpl_dict, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("⬇️ تحميل template.json", data=tpl_bytes,
                               file_name="template.json", mime="application/json")
            st.success("احفظ template.json ثم انتقل للتبويب الثاني للتصحيح")

# =========================
# TAB 2: Grading
# =========================
with tab2:
    st.subheader("1) ارفع template.json")
    tpl_json_file = st.file_uploader("template.json", type=["json"], key="tpl_json_file")

    tpl: Optional[Template] = None
    if tpl_json_file:
        tpl_raw = json.loads(tpl_json_file.getvalue().decode("utf-8"))
        tpl = Template(
            base_w=int(tpl_raw["base_w"]),
            base_h=int(tpl_raw["base_h"]),
            id_roi=tuple(tpl_raw["id_roi"]),
            id_digits=int(tpl_raw["id_digits"]),
            id_rows=int(tpl_raw["id_rows"]),
            q_blocks=[
                QBlock(
                    name=b["name"],
                    roi=tuple(b["roi"]),
                    q_start=int(b["q_start"]),
                    q_end=int(b["q_end"]),
                    rows=int(b["rows"]),
                ) for b in tpl_raw["q_blocks"]
            ],
            choices_default=int(tpl_raw.get("choices_default", 4))
        )
        st.success("✅ Template loaded")

    st.markdown("---")
    st.subheader("2) إعدادات التصحيح")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        dpi_g = st.slider("DPI تحويل PDF (تصحيح)", 80, 220, 140, 10, key="dpi_g")
    with col2:
        choices = st.radio("عدد الخيارات", [4, 5], horizontal=True, index=0)
    with col3:
        start_page = st.number_input("Start page", min_value=1, value=1, step=1)
    with col4:
        end_page = st.number_input("End page", min_value=1, value=50, step=1)

    st.markdown("**نطاقات التصحيح (اختياري)**")
    cA, cB = st.columns([1, 1])
    with cA:
        theory_txt = st.text_input("نطاق النظري (مثال: 1-70 أو 1-40,50-60)", "")
    with cB:
        practical_txt = st.text_input("نطاق العملي (اختياري)", "")

    theory_ranges = parse_ranges(theory_txt)
    practical_ranges = parse_ranges(practical_txt)

    st.markdown("**حساسية القراءة** (لتحسين قراءة الكود والببل)")
    cI, cJ, cK, cL = st.columns([1, 1, 1, 1])
    with cI:
        id_blank_thr = st.slider("ID blank thr", 0.01, 0.40, 0.06, 0.01)
    with cJ:
        id_min_ratio = st.slider("ID min ratio", 1.05, 2.50, 1.10, 0.05)
    with cK:
        ans_blank_thr = st.slider("Ans blank thr", 0.01, 0.40, 0.08, 0.01)
    with cL:
        ans_min_ratio = st.slider("Ans min ratio", 1.05, 2.50, 1.20, 0.05)

    st.markdown("---")
    st.subheader("3) الملفات")
    roster_file = st.file_uploader("Roster (student_code, student_name)", type=["xlsx", "xls", "csv"], key="roster_file")
    key_file = st.file_uploader("Answer Key (PDF صفحة واحدة أو صورة)", type=["pdf", "png", "jpg", "jpeg"], key="key_file")
    sheets_file = st.file_uploader("Student Sheets (PDF متعدد الصفحات أو صور)", type=["pdf", "png", "jpg", "jpeg"], key="sheets_file")

    roster = {}
    if roster_file:
        fn = roster_file.name.lower()
        df = pd.read_csv(roster_file) if fn.endswith(".csv") else pd.read_excel(roster_file)
        df.columns = [c.strip().lower() for c in df.columns]
        if "student_code" not in df.columns or "student_name" not in df.columns:
            st.error("Roster لازم يحتوي عمودين: student_code و student_name")
            st.stop()

        # Important: keep leading zeros
        df["student_code"] = df["student_code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        # Default to 4 digits if your code length is 4; change if needed:
        df["student_code"] = df["student_code"].apply(lambda x: x.zfill(4))
        df["student_name"] = df["student_name"].astype(str).str.strip()
        roster = dict(zip(df["student_code"], df["student_name"]))
        st.success(f"✅ Loaded roster: {len(roster)}")

    debug = st.checkbox("Debug: عرض قصّ ID + بلوكات أول ورقة", True)

    disabled_run = (tpl is None)
    if st.button("🚀 ابدأ التصحيح", disabled=disabled_run):
        if tpl is None:
            st.error("ارفع template.json أولاً")
            st.stop()
        if not (roster_file and key_file and sheets_file):
            st.error("ارفع Roster + AnswerKey + Student Sheets")
            st.stop()
        if end_page < start_page:
            st.error("End page لازم >= Start page")
            st.stop()

        # ---- Answer Key page 1
        key_pages = load_pages(key_file.getvalue(), key_file.name, dpi=dpi_g, first_page=1, last_page=1)
        key_img = key_pages[0]
        key_thr = preprocess(pil_to_bgr(key_img))

        kw, kh = key_img.size
        tpl_key = template_for_page(tpl, kw, kh)
        use_choices = int(choices) if choices else tpl.choices_default

        key_ans = read_answers(key_thr, tpl_key, use_choices, ans_blank_thr, ans_min_ratio)

        # ---- Student pages
        if sheets_file.name.lower().endswith(".pdf"):
            pages = load_pages(
                sheets_file.getvalue(), sheets_file.name,
                dpi=dpi_g, first_page=int(start_page), last_page=int(end_page)
            )
        else:
            pages = [Image.open(io.BytesIO(sheets_file.getvalue()))]

        total = len(pages)
        prog = st.progress(0)
        results = []

        for i, pg in enumerate(pages, 1):
            sheet_index = int(start_page) + i - 1

            thr = preprocess(pil_to_bgr(pg))
            pw, ph = pg.size
            tpl_p = template_for_page(tpl, pw, ph)

            code, code_status = read_student_code(thr, tpl_p, id_blank_thr, id_min_ratio)
            name = roster.get(code, "") if "?" not in code else ""

            stu_ans = read_answers(thr, tpl_p, use_choices, ans_blank_thr, ans_min_ratio)
            score = compute_score(key_ans, stu_ans, theory_ranges, practical_ranges)

            if debug and i == 1:
                st.markdown("### 🔎 Debug (First Student Page)")
                x, y, w, h = tpl_p.id_roi
                st.image(thr[y:y + h, x:x + w], caption=f"ID ROI | code={code} | status={code_status}", clamp=True)
                for b in tpl_p.q_blocks:
                    x, y, w, h = b.roi
                    st.image(thr[y:y + h, x:x + w], caption=f"{b.name}: Q{b.q_start}-{b.q_end}", clamp=True)

                st.write("Key sample:", {k: key_ans[k] for k in sorted(key_ans)[:10]})
                st.write("Student sample:", {k: stu_ans[k] for k in sorted(stu_ans)[:10]})

            results.append({
                "sheet_index": sheet_index,
                "student_code": code,
                "student_name": name,
                "score": int(score),
                "id_status": ",".join(code_status)
            })

            prog.progress(int(i / total * 100))

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
