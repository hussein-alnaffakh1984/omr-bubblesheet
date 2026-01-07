# app.py
# ✅ OMR Bubble Sheet (Cloud-Stable) — FINAL Professional UI
# - No nested columns (fixes StreamlitAPIException)
# - No drawable canvas (avoids white/blank background issue on Streamlit Cloud)
# - ROI selection by TWO clicks (top-left then bottom-right) on a scaled preview
# - Save ID ROI + multiple Q Blocks
# - Reads: student_code + answers, outputs: sheet_index, student_code, student_name, score, total_questions
# - Supports 4 or 5 choices
# - Supports grading a subset by ranges (theory/practical). If both empty => grade all questions in blocks.
#
# Requirements (requirements.txt):
# streamlit==1.40.0
# streamlit-image-coordinates==0.1.6
# numpy==1.26.4
# pandas==2.1.4
# Pillow==10.4.0
# opencv-python-headless==4.8.1.78
# pdf2image==1.17.0
# openpyxl==3.1.2
#
# packages.txt (for PDF):
# poppler-utils

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
from streamlit_image_coordinates import streamlit_image_coordinates


# =========================
# Data Models
# =========================
@dataclass
class IdConfig:
    x: int
    y: int
    w: int
    h: int
    digits: int = 4
    rows: int = 10


@dataclass
class QBlockConfig:
    x: int
    y: int
    w: int
    h: int
    start_q: int
    end_q: int
    rows: int


@dataclass
class TemplateConfig:
    page_w: int
    page_h: int
    id_cfg: Optional[IdConfig] = None
    q_blocks: List[QBlockConfig] = None

    def to_dict(self):
        d = asdict(self)
        if d.get("q_blocks") is None:
            d["q_blocks"] = []
        return d

    @staticmethod
    def from_dict(d: dict):
        id_cfg = d.get("id_cfg")
        if id_cfg is not None:
            id_cfg = IdConfig(**id_cfg)
        q_blocks = [QBlockConfig(**qb) for qb in d.get("q_blocks", [])]
        return TemplateConfig(
            page_w=int(d["page_w"]),
            page_h=int(d["page_h"]),
            id_cfg=id_cfg,
            q_blocks=q_blocks,
        )


# =========================
# Image + Parsing Utilities
# =========================
def load_pages(file_bytes: bytes, filename: str) -> List[Image.Image]:
    name = filename.lower()
    if name.endswith(".pdf"):
        return convert_from_bytes(file_bytes)
    return [Image.open(io.BytesIO(file_bytes))]


def pil_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        10,
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
        return False
    return any(a <= q <= b for a, b in ranges)


def should_count_question(q: int, theory_ranges, practical_ranges) -> bool:
    # If both empty => grade everything
    if not theory_ranges and not practical_ranges:
        return True
    return in_ranges(q, theory_ranges) or in_ranges(q, practical_ranges)


def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def rect_from_two_clicks(p1, p2, scale: float) -> Tuple[int, int, int, int]:
    # p1,p2 are in resized coords; convert back to original by /scale
    x1, y1 = int(p1["x"] / scale), int(p1["y"] / scale)
    x2, y2 = int(p2["x"] / scale), int(p2["y"] / scale)
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return x, y, w, h


def draw_rect(preview: Image.Image, rect_resized: Tuple[int, int, int, int]) -> Image.Image:
    arr = np.array(preview.copy())
    x, y, w, h = rect_resized
    x2, y2 = x + w, y + h
    t = max(2, int(min(preview.size) * 0.003))
    cv2.rectangle(arr, (x, y), (x2, y2), (255, 0, 0), t)
    return Image.fromarray(arr)


# =========================
# OMR Readers
# =========================
def read_student_code(thr: np.ndarray, id_cfg: IdConfig) -> str:
    H, W = thr.shape[:2]
    x, y, w, h = clamp_roi(id_cfg.x, id_cfg.y, id_cfg.w, id_cfg.h, W, H)
    roi = thr[y : y + h, x : x + w]

    rows = int(id_cfg.rows)
    cols = int(id_cfg.digits)
    ch = max(1, h // rows)
    cw = max(1, w // cols)

    cell_area = max(1, ch * cw)
    min_fill = int(cell_area * 0.12)
    min_ratio = 1.25

    digits = []
    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw]
            scores.append((str(r), score_cell(cell)))
        d, stt = pick_one(scores, min_fill, min_ratio)
        digits.append("" if d in ["?", "!"] else d)

    return "".join(digits)


def read_qblock_answers(thr: np.ndarray, qb: QBlockConfig, choices: int) -> Dict[int, Tuple[str, str]]:
    H, W = thr.shape[:2]
    x, y, w, h = clamp_roi(qb.x, qb.y, qb.w, qb.h, W, H)
    roi = thr[y : y + h, x : x + w]

    letters = "ABCDE"[:choices]
    q_count = qb.end_q - qb.start_q + 1
    rows = int(qb.rows) if qb.rows and qb.rows > 0 else q_count
    if abs(rows - q_count) > max(2, int(q_count * 0.25)):
        rows = q_count

    rh = max(1, h // rows)
    cw = max(1, w // choices)

    cell_area = max(1, rh * cw)
    min_fill = int(cell_area * 0.10)
    min_ratio = 1.25

    out = {}
    q = qb.start_q
    for r in range(rows):
        if q > qb.end_q:
            break
        scores = []
        for c in range(choices):
            cell = roi[r * rh : (r + 1) * rh, c * cw : (c + 1) * cw]
            scores.append((letters[c], score_cell(cell)))
        a, stt = pick_one(scores, min_fill, min_ratio)
        out[q] = (a, stt)
        q += 1

    return out


# =========================
# Streamlit State
# =========================
def init_state():
    if "cfg" not in st.session_state:
        st.session_state.cfg = None
    if "tpl_img" not in st.session_state:
        st.session_state.tpl_img = None
    if "clicks" not in st.session_state:
        st.session_state.clicks = []  # keep last two clicks only
    if "rect_orig" not in st.session_state:
        st.session_state.rect_orig = None
    if "rect_resized" not in st.session_state:
        st.session_state.rect_resized = None
    if "results_bytes" not in st.session_state:
        st.session_state.results_bytes = None
    if "results_df" not in st.session_state:
        st.session_state.results_df = None


init_state()


# =========================
# UI (No nested columns)
# =========================
st.set_page_config(page_title="OMR Pro (Cloud)", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2.0rem; }
      .card{
        border:1px solid rgba(49,51,63,.18);
        border-radius:16px;
        padding:14px 14px 10px 14px;
        background:rgba(250,250,252,.7);
        margin-bottom:12px;
      }
      .tiny{ font-size:.86rem; opacity:.85; }
      .pill{
        display:inline-block; padding:4px 10px; border-radius:999px;
        border:1px solid rgba(49,51,63,.25);
        font-size:.85rem; margin-left:6px;
      }
      .stButton button, .stDownloadButton button { border-radius: 12px; padding: .6rem .95rem; }
      .stTextInput input { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("✅ OMR Bubble Sheet — النسخة النهائية (واجهة احترافية)")
st.caption("تحديد المناطق بالنقر مرتين (مثل Remark). بدون Canvas لضمان الاستقرار على Streamlit Cloud.")

# Top-level layout (ONLY ONE columns level)
col_left, col_right = st.columns([1.45, 0.55])

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("⚙️ إعدادات الامتحان")
    choices = st.radio("عدد الخيارات", [4, 5], horizontal=True)

    st.markdown("**نطاق التصحيح (اختياري)**")
    theory_txt = st.text_input("نطاق النظري (مثال: 1-70 أو 1-40)", "")
    practical_txt = st.text_input("نطاق العملي (مثال: 1-25)", "")
    theory_ranges = parse_ranges(theory_txt)
    practical_ranges = parse_ranges(practical_txt)
    st.caption("إذا تركت الاثنين فارغين → سيتم تصحيح كل الأسئلة في البلوكات.")

    st.divider()
    st.markdown("### 🧹 أدوات")
    if st.button("مسح النقرات"):
        st.session_state.clicks = []
        st.session_state.rect_orig = None
        st.session_state.rect_resized = None
    if st.button("Reset Config"):
        if st.session_state.tpl_img is not None:
            W, H = st.session_state.tpl_img.size
            st.session_state.cfg = TemplateConfig(page_w=W, page_h=H, id_cfg=None, q_blocks=[])
        else:
            st.session_state.cfg = None
        st.session_state.results_bytes = None
        st.session_state.results_df = None

# -------------------------
# LEFT: Template + ROI Selection
# -------------------------
with col_left:
    st.markdown(
        '<div class="card"><h3 style="margin:0">1) Template + تحديد المناطق</h3>'
        '<div class="tiny">ارفع Template (PDF صفحة واحدة أو صورة)، ثم انقر مرتين لتحديد المستطيل.</div></div>',
        unsafe_allow_html=True,
    )

    tpl_file = st.file_uploader("📄 Template", type=["pdf", "png", "jpg", "jpeg"])

    # Config upload/download (NO nested columns: use simple flow)
    st.markdown('<div class="card"><h4 style="margin:0">Config (اختياري)</h4><div class="tiny">ارفع config.json أو نزّله بعد التحديد.</div></div>', unsafe_allow_html=True)
    cfg_up = st.file_uploader("⬆️ رفع config.json", type=["json"])
    if cfg_up:
        try:
            cfg_obj = json.loads(cfg_up.getvalue().decode("utf-8"))
            st.session_state.cfg = TemplateConfig.from_dict(cfg_obj)
            st.success("✅ تم تحميل config.json")
        except Exception as e:
            st.error(f"فشل تحميل JSON: {e}")

    # Load template
    if tpl_file:
        pages = load_pages(tpl_file.getvalue(), tpl_file.name)
        tpl_img = pil_rgb(pages[0])
        st.session_state.tpl_img = tpl_img
        W, H = tpl_img.size
        if st.session_state.cfg is None:
            st.session_state.cfg = TemplateConfig(page_w=W, page_h=H, id_cfg=None, q_blocks=[])
        else:
            st.session_state.cfg.page_w = W
            st.session_state.cfg.page_h = H

    if st.session_state.tpl_img is None:
        st.info("⬆️ ارفع Template أولاً.")
    else:
        tpl_img = st.session_state.tpl_img
        cfg: TemplateConfig = st.session_state.cfg
        page_w, page_h = tpl_img.size

        # Download config if exists
        if cfg is not None:
            cfg_bytes = json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("⬇️ تنزيل config.json", cfg_bytes, file_name="config.json", mime="application/json")

        st.markdown(
            '<div class="card"><h4 style="margin:0">معاينة + تحديد</h4>'
            '<div class="tiny">اضغط مرتين: (1) أعلى يسار، (2) أسفل يمين. ثم احفظ كـ ID أو Q Block.</div></div>',
            unsafe_allow_html=True,
        )

        # Preview size controls (no nested columns)
        fit = st.checkbox("📌 Fit to screen", value=True)
        preset = st.select_slider("حجم المعاينة", options=["Small", "Medium", "Large"], value="Medium")
        preset_w = {"Small": 900, "Medium": 1100, "Large": 1400}[preset]
        zoom_w = preset_w if fit else st.slider("عرض المعاينة (px)", 700, 1800, preset_w, 50)

        scale = zoom_w / page_w
        resized_h = int(page_h * scale)
        preview = tpl_img.resize((zoom_w, resized_h))

        mode = st.radio("وضع التحديد", ["ID (كود الطالب)", "Q Block (أسئلة)"], horizontal=True)

        st.markdown(
            f'<span class="pill">Page: {page_w}×{page_h}</span>'
            f'<span class="pill">Preview: {zoom_w}×{resized_h}</span>',
            unsafe_allow_html=True,
        )

        # Capture click (works on Cloud)
        click = streamlit_image_coordinates(preview, key="tpl_click", width=zoom_w)

        if click:
            st.session_state.clicks.append(click)
            if len(st.session_state.clicks) > 2:
                st.session_state.clicks = st.session_state.clicks[-2:]

        if len(st.session_state.clicks) == 2:
            p1, p2 = st.session_state.clicks
            x, y, w, h = rect_from_two_clicks(p1, p2, scale)
            x, y, w, h = clamp_roi(x, y, w, h, page_w, page_h)
            st.session_state.rect_orig = (x, y, w, h)

            rx, ry, rw, rh = int(x * scale), int(y * scale), int(w * scale), int(h * scale)
            st.session_state.rect_resized = (rx, ry, rw, rh)

        # Show preview with rectangle
        shown = preview
        if st.session_state.rect_resized:
            shown = draw_rect(preview, st.session_state.rect_resized)
        st.image(shown, use_container_width=True)

        if st.session_state.rect_orig:
            x, y, w, h = st.session_state.rect_orig
            st.success(f"✅ مستطيل جاهز: x={x}, y={y}, w={w}, h={h}")

            crop = tpl_img.crop((x, y, x + w, y + h))
            st.image(crop, caption="Crop (الأصل)", use_container_width=True)

            if mode.startswith("ID"):
                st.markdown('<div class="card"><h4 style="margin:0">حفظ ID ROI</h4></div>', unsafe_allow_html=True)
                digits = st.number_input("خانات الكود", 2, 12, 4, 1)
                rows = st.number_input("صفوف الأرقام", 5, 15, 10, 1)

                if st.button("✅ حفظ ID ROI", type="primary"):
                    cfg.id_cfg = IdConfig(x=x, y=y, w=w, h=h, digits=int(digits), rows=int(rows))
                    st.session_state.clicks = []
                    st.session_state.rect_orig = None
                    st.session_state.rect_resized = None
                    st.success("تم حفظ ID ROI ✅")

            else:
                st.markdown('<div class="card"><h4 style="margin:0">إضافة Q Block</h4>'
                            '<div class="tiny">حدد فقط أول سؤال + عدد الأسئلة. النهاية تُحسب تلقائياً.</div></div>',
                            unsafe_allow_html=True)
                start_q = st.number_input("أول سؤال في هذا البلوك", 1, 999, 1, 1)
                n_q = st.number_input("عدد الأسئلة داخل البلوك", 1, 999, 20, 1)
                rows = st.number_input("Rows (عادة = عدد الأسئلة)", 1, 999, int(n_q), 1)
                end_q = int(start_q) + int(n_q) - 1
                st.info(f"سيتم حفظ البلوك: Q{start_q} → Q{end_q}")

                if st.button("✅ إضافة البلوك", type="primary"):
                    cfg.q_blocks.append(QBlockConfig(x=x, y=y, w=w, h=h, start_q=int(start_q), end_q=int(end_q), rows=int(rows)))
                    st.session_state.clicks = []
                    st.session_state.rect_orig = None
                    st.session_state.rect_resized = None
                    st.success("تمت إضافة البلوك ✅")

        # Summary (no nested columns: show sequential)
        st.markdown('<div class="card"><h3 style="margin:0">ملخص التحديد</h3></div>', unsafe_allow_html=True)
        if cfg.id_cfg:
            st.markdown("**ID ROI:**")
            st.write(cfg.id_cfg)
        else:
            st.warning("لم يتم تحديد ID ROI بعد.")

        if cfg.q_blocks:
            st.markdown("**Q Blocks:**")
            dfb = pd.DataFrame([asdict(b) for b in cfg.q_blocks])
            st.dataframe(dfb, use_container_width=True, hide_index=True)

            st.markdown('<div class="card"><h4 style="margin:0">إدارة البلوكات</h4></div>', unsafe_allow_html=True)
            idx_del = st.number_input("حذف بلوك رقم", 1, len(cfg.q_blocks), 1, 1)
            if st.button("🗑️ حذف البلوك المحدد"):
                del cfg.q_blocks[int(idx_del) - 1]
                st.success("تم حذف البلوك ✅")
            if st.button("↕️ عكس ترتيب البلوكات"):
                cfg.q_blocks = list(reversed(cfg.q_blocks))
                st.success("تم عكس الترتيب ✅")
        else:
            st.info("لم تتم إضافة Q Blocks بعد.")


# -------------------------
# RIGHT: Upload files + Run grading
# -------------------------
with col_right:
    st.markdown(
        '<div class="card"><h3 style="margin:0">2) ملفات التصحيح</h3>'
        '<div class="tiny">ارفع roster + answer key + أوراق الطلاب.</div></div>',
        unsafe_allow_html=True,
    )

    roster_file = st.file_uploader("📋 Roster (student_code, student_name)", type=["xlsx", "xls", "csv"])
    key_file = st.file_uploader("🧾 Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    sheets_file = st.file_uploader("🗂️ أوراق الطلاب (PDF متعدد/صور)", type=["pdf", "png", "jpg", "jpeg"])

    st.markdown(
        '<div class="card"><h3 style="margin:0">3) تشغيل التصحيح</h3>'
        '<div class="tiny">النتيجة: درجة واحدة لكل ورقة (sheet_index + code + name + score).</div></div>',
        unsafe_allow_html=True,
    )

    run = st.button("🚀 ابدأ التصحيح الآن", type="primary")

    if st.session_state.results_bytes is not None:
        st.download_button(
            "⬇️ تنزيل results.xlsx",
            st.session_state.results_bytes,
            file_name="results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# =========================
# Run grading
# =========================
if run:
    cfg: TemplateConfig = st.session_state.cfg

    if st.session_state.tpl_img is None or cfg is None:
        st.error("ارفع Template وحدد المناطق أولاً.")
        st.stop()

    if cfg.id_cfg is None:
        st.error("حدد منطقة كود الطالب (ID ROI) أولاً.")
        st.stop()

    if not cfg.q_blocks:
        st.error("أضف Q Block واحد على الأقل.")
        st.stop()

    if not (roster_file and key_file and sheets_file):
        st.error("ارفع Roster + Answer Key + Student Sheets.")
        st.stop()

    # Load roster
    try:
        if roster_file.name.lower().endswith((".xlsx", ".xls")):
            df_roster = pd.read_excel(roster_file)
        else:
            df_roster = pd.read_csv(roster_file)

        df_roster.columns = [c.strip().lower() for c in df_roster.columns]
        if "student_code" not in df_roster.columns or "student_name" not in df_roster.columns:
            st.error("Roster لازم يحتوي عمودين: student_code و student_name")
            st.stop()

        roster = dict(zip(df_roster["student_code"].astype(str), df_roster["student_name"].astype(str)))
        st.success(f"✅ تم تحميل roster: {len(roster)} طالب")
    except Exception as e:
        st.error(f"فشل قراءة Roster: {e}")
        st.stop()

    # Load Answer Key
    try:
        key_pages = load_pages(key_file.getvalue(), key_file.name)
        key_img = pil_rgb(key_pages[0])
        key_thr = preprocess(pil_to_cv(key_img))

        key_answers: Dict[int, Tuple[str, str]] = {}
        for qb in cfg.q_blocks:
            key_answers.update(read_qblock_answers(key_thr, qb, choices))
        if not key_answers:
            st.error("لم يتم استخراج أي إجابات من Answer Key. تأكد من Q Blocks.")
            st.stop()
        st.success(f"✅ تم استخراج Answer Key: {len(key_answers)} سؤال")
    except Exception as e:
        st.error(f"فشل قراءة Answer Key: {e}")
        st.stop()

    # Load student sheets
    try:
        pages = load_pages(sheets_file.getvalue(), sheets_file.name)
        total_pages = len(pages)
        st.info(f"عدد صفحات الطلاب: {total_pages}")
    except Exception as e:
        st.error(f"فشل قراءة أوراق الطلاب: {e}")
        st.stop()

    prog = st.progress(0)
    status = st.empty()

    results = []
    for idx, pg in enumerate(pages, 1):
        status.write(f"تصحيح الورقة {idx}/{total_pages} ...")
        img = pil_rgb(pg)
        thr = preprocess(pil_to_cv(img))

        code = read_student_code(thr, cfg.id_cfg)
        name = roster.get(str(code), "")

        stu_answers: Dict[int, Tuple[str, str]] = {}
        for qb in cfg.q_blocks:
            stu_answers.update(read_qblock_answers(thr, qb, choices))

        score = 0
        total_counted = 0
        for q, (ka, _) in key_answers.items():
            if not should_count_question(q, theory_ranges, practical_ranges):
                continue
            total_counted += 1
            sa, _ = stu_answers.get(q, ("?", "MISSING"))
            if sa == ka:
                score += 1

        results.append(
            {
                "sheet_index": idx,
                "student_code": code,
                "student_name": name,
                "score": int(score),
                "total_questions": int(total_counted),
            }
        )

        prog.progress(int(idx / total_pages * 100))

    out_df = pd.DataFrame(results)
    st.success("✅ انتهى التصحيح")
    st.dataframe(out_df, use_container_width=True, hide_index=True)

    buf = io.BytesIO()
    out_df.to_excel(buf, index=False)
    st.session_state.results_bytes = buf.getvalue()
    st.session_state.results_df = out_df

    st.download_button(
        "⬇️ تنزيل results.xlsx",
        st.session_state.results_bytes,
        file_name="results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
