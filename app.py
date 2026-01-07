import io
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# =========================
# Data models
# =========================
@dataclass
class IdConfig:
    x: int
    y: int
    w: int
    h: int
    digits: int = 4          # عدد خانات كود الطالب
    rows: int = 10           # صفوف الأرقام 0-9


@dataclass
class QBlockConfig:
    x: int
    y: int
    w: int
    h: int
    start_q: int
    end_q: int
    rows: int               # عدد صفوف الأسئلة داخل هذا البلوك (عادة = end_q-start_q+1)


@dataclass
class TemplateConfig:
    # أبعاد صفحة النموذج الأصلية (قبل أي تصغير)
    page_w: int
    page_h: int
    id_cfg: Optional[IdConfig] = None
    q_blocks: List[QBlockConfig] = None

    def to_dict(self):
        d = asdict(self)
        # dataclass inside
        if self.id_cfg is None:
            d["id_cfg"] = None
        if self.q_blocks is None:
            d["q_blocks"] = []
        return d

    @staticmethod
    def from_dict(d):
        id_cfg = d.get("id_cfg")
        if id_cfg is not None:
            id_cfg = IdConfig(**id_cfg)
        q_blocks = [QBlockConfig(**qb) for qb in d.get("q_blocks", [])]
        return TemplateConfig(
            page_w=int(d["page_w"]),
            page_h=int(d["page_h"]),
            id_cfg=id_cfg,
            q_blocks=q_blocks
        )


# =========================
# Helpers
# =========================
def load_pages(file_bytes: bytes, filename: str) -> List[Image.Image]:
    name = filename.lower()
    if name.endswith(".pdf"):
        # first page only for template, but for student sheets may be multi-pages
        return convert_from_bytes(file_bytes)
    return [Image.open(io.BytesIO(file_bytes))]


def pil_rgb(img: Image.Image) -> Image.Image:
    # IMPORTANT: must be PIL Image, not numpy
    return img.convert("RGB")


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
    # scores: list[(choice, filled_pixels)]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0

    if top_s < min_fill:
        return "?", "BLANK"

    # double/ambiguous
    if second_s > 0 and (top_s / (second_s + 1e-6)) < min_ratio:
        return "!", "DOUBLE"

    return top_c, "OK"


def parse_ranges(txt: str) -> List[Tuple[int, int]]:
    # examples: "1-40" or "1-40, 55-60, 70"
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
        return True  # إذا ما حددت نطاق، يعني صحّح الكل
    return any(a <= q <= b for a, b in ranges)


def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def scale_rect_from_canvas_to_page(rect, canvas_w, canvas_h, page_w, page_h):
    # rect from drawable_canvas uses float and includes left/top/width/height
    sx = page_w / canvas_w
    sy = page_h / canvas_h
    x = int(rect["left"] * sx)
    y = int(rect["top"] * sy)
    w = int(rect["width"] * sx)
    h = int(rect["height"] * sy)
    return x, y, w, h


# =========================
# OMR reading
# =========================
def read_student_code(thr: np.ndarray, id_cfg: IdConfig) -> str:
    H, W = thr.shape[:2]
    x, y, w, h = clamp_roi(id_cfg.x, id_cfg.y, id_cfg.w, id_cfg.h, W, H)
    roi = thr[y:y + h, x:x + w]

    rows = id_cfg.rows
    cols = id_cfg.digits

    ch = h // rows
    cw = w // cols

    # dynamic thresholds based on area
    cell_area = max(1, ch * cw)
    min_fill = int(cell_area * 0.12)   # adjust if needed
    min_ratio = 1.25

    digits = []
    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            scores.append((str(r), score_cell(cell)))
        d, stt = pick_one(scores, min_fill=min_fill, min_ratio=min_ratio)
        digits.append("" if d in ["?", "!"] else d)

    code = "".join(digits)
    return code


def read_answers(thr: np.ndarray, qblock: QBlockConfig, choices: int) -> Dict[int, Tuple[str, str]]:
    H, W = thr.shape[:2]
    x, y, w, h = clamp_roi(qblock.x, qblock.y, qblock.w, qblock.h, W, H)
    roi = thr[y:y + h, x:x + w]

    letters = "ABCDE"[:choices]
    rows = qblock.rows
    q_count = qblock.end_q - qblock.start_q + 1

    # إذا المستخدم كتب rows غلط، نحاول نضبطه
    if rows <= 0:
        rows = q_count
    # إذا rows أكبر بكثير، نخليه q_count
    if abs(rows - q_count) > max(2, int(q_count * 0.25)):
        rows = q_count

    rh = max(1, h // rows)
    cw = max(1, w // choices)

    # dynamic thresholds
    cell_area = max(1, rh * cw)
    min_fill = int(cell_area * 0.10)  # adjust if needed
    min_ratio = 1.25

    out = {}
    q = qblock.start_q
    for r in range(rows):
        if q > qblock.end_q:
            break
        scores = []
        for c in range(choices):
            cell = roi[r * rh:(r + 1) * rh, c * cw:(c + 1) * cw]
            scores.append((letters[c], score_cell(cell)))
        a, stt = pick_one(scores, min_fill=min_fill, min_ratio=min_ratio)
        out[q] = (a, stt)
        q += 1

    return out


# =========================
# UI
# =========================
st.set_page_config(page_title="OMR Bubble Sheet (Remark Style)", layout="wide")
st.title("✅ OMR Bubble Sheet — واجهة مثل Remark (تحديد يدوي + تصحيح + Excel)")

# Session init
if "cfg" not in st.session_state:
    st.session_state.cfg = None
if "template_img" not in st.session_state:
    st.session_state.template_img = None
if "template_page_wh" not in st.session_state:
    st.session_state.template_page_wh = None


# =========================
# Step 1: Upload template
# =========================
st.header("1) رفع نموذج الورقة (Template)")

tpl_file = st.file_uploader("ارفع PDF صفحة واحدة أو صورة (PNG/JPG) للنموذج", type=["pdf", "png", "jpg", "jpeg"])

colL, colR = st.columns([1.2, 0.8], vertical_alignment="top")

with colR:
    st.subheader("⚙️ إعدادات عامة")
    choices = st.radio("عدد الخيارات", [4, 5], horizontal=True)
    canvas_w = st.slider("Canvas width (كلما زاد أوضح لكن أبطأ)", 800, 1800, 1250, 50)

    st.subheader("نطاق التصحيح (اختياري)")
    correct_ranges_txt = st.text_input("مثال: 1-40 أو 1-70 أو 1-70, 1-25 (إذا عملي منفصل اكتب أدناه)")
    practical_ranges_txt = st.text_input("نطاق عملي (اختياري)", value="")

    correct_ranges = parse_ranges(correct_ranges_txt)
    practical_ranges = parse_ranges(practical_ranges_txt)

    st.divider()
    st.subheader("ملفات التصحيح")
    roster_file = st.file_uploader("Roster Excel/CSV: student_code, student_name", type=["xlsx", "xls", "csv"])
    key_file = st.file_uploader("Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    sheets_file = st.file_uploader("أوراق الطلاب (PDF متعدد الصفحات أو صور)", type=["pdf", "png", "jpg", "jpeg"])

    st.divider()
    st.subheader("حفظ/تحميل إعدادات النموذج")
    cfg_json_up = st.file_uploader("رفع config.json (اختياري)", type=["json"], key="cfg_json_up")

    if cfg_json_up:
        try:
            cfg_obj = json.loads(cfg_json_up.getvalue().decode("utf-8"))
            st.session_state.cfg = TemplateConfig.from_dict(cfg_obj)
            st.success("✅ تم تحميل الإعدادات من JSON")
        except Exception as e:
            st.error(f"فشل تحميل JSON: {e}")

    if st.session_state.cfg:
        cfg_bytes = json.dumps(st.session_state.cfg.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("⬇️ تنزيل config.json", cfg_bytes, file_name="config.json", mime="application/json")


with colL:
    st.subheader("واجهة التحديد (ارسم مستطيل)")

    if not tpl_file and not st.session_state.template_img:
        st.info("ارفع نموذج الورقة أولاً.")
    else:
        if tpl_file:
            tpl_pages = load_pages(tpl_file.getvalue(), tpl_file.name)
            template_img = pil_rgb(tpl_pages[0])
            st.session_state.template_img = template_img
            st.session_state.template_page_wh = (template_img.size[0], template_img.size[1])

            # init cfg if none
            if st.session_state.cfg is None:
                st.session_state.cfg = TemplateConfig(
                    page_w=template_img.size[0],
                    page_h=template_img.size[1],
                    id_cfg=None,
                    q_blocks=[]
                )

        template_img = st.session_state.template_img
        page_w, page_h = st.session_state.template_page_wh

        # resize for canvas preview
        scale = canvas_w / page_w
        canvas_h = int(page_h * scale)
        preview = template_img.resize((canvas_w, canvas_h))

        mode = st.radio("ماذا تحدد الآن؟", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], horizontal=True)

        # IMPORTANT: background_image must be PIL Image (RGB)
        canvas_result = st_canvas(
            fill_color="rgba(255,0,0,0.18)",
            stroke_width=3,
            stroke_color="red",
            background_color="rgba(255,255,255,1)",
            background_image=preview.convert("RGB"),
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="rect",
            key="canvas_main"
        )

        st.caption("💡 نصيحة: ارسم مستطيل كبير يغطي كل الببل الخاص بالأسئلة داخل البلوك بدون الهيدر/النص.")

        # Parse objects
        objs = []
        if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
            objs = canvas_result.json_data["objects"] or []

        # Show objects list + assign
        if objs:
            st.write(f"عدد المستطيلات المرسومة: **{len(objs)}**")
            last = objs[-1]
            x, y, w, h = scale_rect_from_canvas_to_page(last, canvas_w, canvas_h, page_w, page_h)
            st.code(f"آخر مستطيل (على الصفحة الأصلية): x={x}, y={y}, w={w}, h={h}")

            if mode.startswith("ID ROI"):
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    id_digits = st.number_input("عدد خانات الكود", min_value=2, max_value=12, value=4, step=1)
                with c2:
                    id_rows = st.number_input("عدد صفوف أرقام الكود (عادة 10)", min_value=5, max_value=15, value=10, step=1)
                with c3:
                    if st.button("✅ اعتماد هذا المستطيل كـ ID ROI"):
                        st.session_state.cfg.id_cfg = IdConfig(x=x, y=y, w=w, h=h, digits=int(id_digits), rows=int(id_rows))
                        st.success("تم حفظ ID ROI داخل الإعدادات.")

            else:
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    start_q = st.number_input("Start Q", min_value=1, max_value=500, value=1, step=1)
                with c2:
                    end_q = st.number_input("End Q", min_value=1, max_value=500, value=20, step=1)
                with c3:
                    rows = st.number_input("Rows داخل البلوك", min_value=1, max_value=500, value=(end_q - start_q + 1), step=1)

                if st.button("✅ إضافة هذا المستطيل كـ Q Block"):
                    qb = QBlockConfig(
                        x=x, y=y, w=w, h=h,
                        start_q=int(start_q), end_q=int(end_q),
                        rows=int(rows)
                    )
                    st.session_state.cfg.q_blocks.append(qb)
                    st.success("تمت إضافة Q Block.")

        # show current cfg summary
        if st.session_state.cfg:
            st.divider()
            st.subheader("ملخص الإعدادات الحالية")
            cfg = st.session_state.cfg
            if cfg.id_cfg:
                st.write("**ID ROI:**", cfg.id_cfg)
            else:
                st.warning("لم يتم تحديد ID ROI بعد.")
            st.write(f"**عدد Q Blocks:** {len(cfg.q_blocks)}")
            if cfg.q_blocks:
                df_blocks = pd.DataFrame([asdict(b) for b in cfg.q_blocks])
                st.dataframe(df_blocks, use_container_width=True)

            cA, cB = st.columns([1, 1])
            with cA:
                if st.button("🧹 حذف آخر Q Block"):
                    if cfg.q_blocks:
                        cfg.q_blocks.pop()
                        st.success("تم حذف آخر Q Block.")
            with cB:
                if st.button("♻️ Reset الكل"):
                    st.session_state.cfg = TemplateConfig(page_w=page_w, page_h=page_h, id_cfg=None, q_blocks=[])
                    st.success("تمت إعادة الضبط.")


# =========================
# Step 2: Grading
# =========================
st.header("2) التصحيح وإخراج Excel")

if st.button("🚀 ابدأ التصحيح الآن"):
    if st.session_state.cfg is None:
        st.error("لا توجد إعدادات نموذج (cfg).")
        st.stop()

    cfg: TemplateConfig = st.session_state.cfg

    if cfg.id_cfg is None:
        st.error("حدد ID ROI أولاً (كود الطالب).")
        st.stop()

    if not cfg.q_blocks:
        st.error("أضف على الأقل Q Block واحد للأسئلة.")
        st.stop()

    if not (roster_file and key_file and sheets_file):
        st.error("ارفع Roster + Answer Key + أوراق الطلاب.")
        st.stop()

    # Load roster
    if roster_file.name.lower().endswith((".xlsx", ".xls")):
        df_roster = pd.read_excel(roster_file)
    else:
        df_roster = pd.read_csv(roster_file)

    df_roster.columns = [c.strip().lower() for c in df_roster.columns]
    if "student_code" not in df_roster.columns or "student_name" not in df_roster.columns:
        st.error("Roster لازم يحتوي أعمدة: student_code و student_name")
        st.stop()

    roster = dict(zip(df_roster["student_code"].astype(str), df_roster["student_name"].astype(str)))
    st.success(f"✅ تم تحميل Roster: {len(roster)} طالب")

    # Load answer key page (first page)
    key_pages = load_pages(key_file.getvalue(), key_file.name)
    key_img = pil_rgb(key_pages[0])

    # Preprocess key
    key_thr = preprocess(pil_to_cv(key_img))

    # Read key answers from all blocks
    key_answers: Dict[int, Tuple[str, str]] = {}
    for qb in cfg.q_blocks:
        key_answers.update(read_answers(key_thr, qb, choices))

    # Load student pages
    pages = load_pages(sheets_file.getvalue(), sheets_file.name)
    total_pages = len(pages)

    results = []
    prog = st.progress(0)

    for idx, pg in enumerate(pages, 1):
        pg_img = pil_rgb(pg)
        thr = preprocess(pil_to_cv(pg_img))

        code = read_student_code(thr, cfg.id_cfg)
        name = roster.get(str(code), "")

        # read student answers across all blocks
        stu_answers: Dict[int, Tuple[str, str]] = {}
        for qb in cfg.q_blocks:
            stu_answers.update(read_answers(thr, qb, choices))

        # score
        score = 0
        total_counted = 0

        for q, (ka, kst) in key_answers.items():
            # decide if this question is included
            use_q = in_ranges(q, correct_ranges) or in_ranges(q, practical_ranges)
            if not use_q:
                continue

            sa, sst = stu_answers.get(q, ("?", "MISSING"))
            # count it
            total_counted += 1

            # strict: BLANK/DOUBLE are wrong
            if sa == ka:
                score += 1

        results.append({
            "sheet_index": idx,
            "student_code": code,
            "student_name": name,
            "score": int(score),
            "total_questions": int(total_counted)
        })

        prog.progress(int(idx / total_pages * 100))

    out_df = pd.DataFrame(results)

    # Export Excel
    buf = io.BytesIO()
    out_df.to_excel(buf, index=False)

    st.success("✅ تم إنشاء ملف Excel")
    st.dataframe(out_df.head(30), use_container_width=True)
    st.download_button("⬇️ تحميل results.xlsx", buf.getvalue(), file_name="results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
