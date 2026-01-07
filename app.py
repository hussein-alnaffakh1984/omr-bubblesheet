# app.py
import io
import json
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

# =========================
# Data Models
# =========================
@dataclass
class QBlock:
    x: int
    y: int
    w: int
    h: int
    start_q: int
    end_q: int
    rows: int  # number of questions vertically inside this block

@dataclass
class TemplateConfig:
    template_width: int
    template_height: int

    # Student Code
    id_roi: Tuple[int, int, int, int]  # (x, y, w, h) in template/original coords
    id_digits: int
    id_rows: int  # usually 10 (0..9)

    # Answers blocks
    q_blocks: List[QBlock]

def empty_config(tw: int, th: int) -> TemplateConfig:
    return TemplateConfig(
        template_width=tw,
        template_height=th,
        id_roi=(0, 0, 1, 1),
        id_digits=4,
        id_rows=10,
        q_blocks=[]
    )

# =========================
# Helpers
# =========================
def load_pages(file_bytes: bytes, filename: str, dpi: int = 150) -> List[Image.Image]:
    name = filename.lower()
    if name.endswith(".pdf"):
        return convert_from_bytes(file_bytes, dpi=dpi)
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def preprocess_for_omr(img_bgr: np.ndarray) -> np.ndarray:
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

def pick_one(scores, min_fill: int, min_ratio: float):
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0

    if top_s < min_fill:
        return "?", "BLANK"

    # if close => double/ambiguous
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

def config_to_json(cfg: TemplateConfig) -> str:
    d = asdict(cfg)
    # q_blocks already dict
    return json.dumps(d, ensure_ascii=False, indent=2)

def json_to_config(s: str) -> TemplateConfig:
    d = json.loads(s)
    qbs = [QBlock(**qb) for qb in d.get("q_blocks", [])]
    return TemplateConfig(
        template_width=int(d["template_width"]),
        template_height=int(d["template_height"]),
        id_roi=tuple(d["id_roi"]),
        id_digits=int(d["id_digits"]),
        id_rows=int(d["id_rows"]),
        q_blocks=qbs,
    )

# =========================
# OMR Readers
# =========================
def read_student_code(thr: np.ndarray, cfg: TemplateConfig, min_fill=250, min_ratio=1.25) -> Tuple[str, str]:
    x, y, w, h = cfg.id_roi
    x = max(0, x); y = max(0, y)
    w = max(1, w); h = max(1, h)

    roi = thr[y:y+h, x:x+w]
    if roi.size == 0:
        return "", "ID_ROI_EMPTY"

    rows, cols = cfg.id_rows, cfg.id_digits
    ch = max(1, h // rows)
    cw = max(1, w // cols)

    digits = []
    status_all = "OK"

    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            scores.append((str(r), score_cell(cell)))
        d, stt = pick_one(scores, min_fill, min_ratio)
        if stt != "OK":
            status_all = "WARN"
        digits.append("" if d in ["?", "!"] else d)

    code = "".join(digits)
    # لو طلع فارغ بالكامل
    if not code.strip():
        return "", "ID_NOT_READ"
    return code, status_all

def read_answers(thr: np.ndarray, cfg: TemplateConfig, choices: int, min_fill=180, min_ratio=1.25) -> Dict[int, Tuple[str, str]]:
    letters = "ABCDE"[:choices]
    out: Dict[int, Tuple[str, str]] = {}

    for blk in cfg.q_blocks:
        x, y, w, h = blk.x, blk.y, blk.w, blk.h
        roi = thr[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        rows = max(1, blk.rows)
        rh = max(1, h // rows)
        cw = max(1, w // choices)

        q = blk.start_q
        for r in range(rows):
            if q > blk.end_q:
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
st.set_page_config(page_title="OMR Bubble Sheet (Remark-style)", layout="wide")
st.title("✅ OMR Bubble Sheet — تحديد يدوي مثل Remark + تصحيح + Excel")

# -------------------------
# Session State
# -------------------------
if "cfg" not in st.session_state:
    st.session_state.cfg = None
if "template_img" not in st.session_state:
    st.session_state.template_img = None
if "template_scale" not in st.session_state:
    st.session_state.template_scale = 1.0

# -------------------------
# Sidebar: Exam Settings
# -------------------------
with st.sidebar:
    st.header("⚙️ إعدادات الامتحان")
    choices = st.radio("عدد الخيارات", [4, 5], horizontal=True)
    dpi = st.select_slider("DPI لتحويل PDF (أقل = أسرع)", options=[100, 120, 150, 200], value=150)

    st.markdown("---")
    st.subheader("نطاق التصحيح")
    theory_txt = st.text_input("نطاق النظري (مثال: 1-40)", value="")
    practical_txt = st.text_input("نطاق العملي (اختياري) (مثال: 41-60)", value="")

    st.markdown("---")
    st.subheader("سياسة المشاكل")
    strict = st.checkbox("BLANK/DOUBLE = خطأ", value=True)
    min_fill_id = st.slider("حساسية كود الطالب (min_fill)", 50, 600, 250, 10)
    min_fill_ans = st.slider("حساسية الإجابة (min_fill)", 50, 600, 180, 10)
    min_ratio = st.slider("تمييز خيارين (min_ratio)", 1.05, 2.00, 1.25, 0.01)

theory_ranges = parse_ranges(theory_txt)
practical_ranges = parse_ranges(practical_txt)

# -------------------------
# Step 1: Upload Template
# -------------------------
st.subheader("1) ارفع نموذج ورقة واحدة (Template) لتحديد المناطق عليها")
tpl_file = st.file_uploader("Template (PDF صفحة واحدة أو صورة)", type=["pdf", "png", "jpg", "jpeg"], key="tpl")

colA, colB = st.columns([1.4, 1.0], gap="large")

with colA:
    if tpl_file:
        pages = load_pages(tpl_file.getvalue(), tpl_file.name, dpi=dpi)
        tpl_img = pages[0].convert("RGB")
        st.session_state.template_img = tpl_img

        orig_w, orig_h = tpl_img.size
        st.caption(f"Template size: {orig_w} x {orig_h}")

        # init config if none
        if st.session_state.cfg is None or st.session_state.cfg.template_width != orig_w or st.session_state.cfg.template_height != orig_h:
            st.session_state.cfg = empty_config(orig_w, orig_h)

        # display size for drawing
        canvas_w = st.slider("Canvas width (للرسم فقط)", 700, 1400, 1100, 50)
        disp_h = int(orig_h * (canvas_w / orig_w))
        disp_img = tpl_img.resize((canvas_w, disp_h))

        scale = orig_w / canvas_w
        st.session_state.template_scale = scale

        # Select tool
        tool = st.radio("ماذا تحدد الآن؟", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], horizontal=True)

        # Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.15)",
            stroke_width=3,
            stroke_color="#ff0000",
            background_image=disp_img,
            update_streamlit=False,   # speed
            realtime_update=False,    # speed
            width=canvas_w,
            height=disp_h,
            drawing_mode="rect",
            key="canvas_main",
        )

        def rect_to_orig(obj) -> Tuple[int, int, int, int]:
            x = int(obj["left"] * scale)
            y = int(obj["top"] * scale)
            w = int(obj["width"] * scale)
            h = int(obj["height"] * scale)
            # pad small safety
            pad = 2
            x = max(0, x + pad)
            y = max(0, y + pad)
            w = max(1, w - 2 * pad)
            h = max(1, h - 2 * pad)
            return x, y, w, h

        # Buttons
        btn1, btn2, btn3 = st.columns([1, 1, 1])
        with btn1:
            add_btn = st.button("➕ حفظ آخر مستطيل", use_container_width=True)
        with btn2:
            undo_btn = st.button("↩️ مسح آخر عنصر", use_container_width=True)
        with btn3:
            reset_btn = st.button("🧹 Reset الكل", use_container_width=True)

        # Reset
        if reset_btn:
            st.session_state.cfg = empty_config(orig_w, orig_h)
            st.rerun()

        # Undo
        if undo_btn:
            if tool.startswith("ID"):
                st.session_state.cfg.id_roi = (0, 0, 1, 1)
            else:
                if st.session_state.cfg.q_blocks:
                    st.session_state.cfg.q_blocks.pop()
            st.rerun()

        # Add
        if add_btn:
            if canvas_result and canvas_result.json_data and canvas_result.json_data.get("objects"):
                last = canvas_result.json_data["objects"][-1]
                x, y, w, h = rect_to_orig(last)

                if tool.startswith("ID"):
                    st.session_state.cfg.id_roi = (x, y, w, h)
                    st.success("✅ تم حفظ ID ROI")
                else:
                    st.session_state.cfg.q_blocks.append(QBlock(x=x, y=y, w=w, h=h, start_q=1, end_q=20, rows=20))
                    st.success("✅ تم إضافة Q Block (عدّل start/end/rows من اليمين)")
                st.rerun()
            else:
                st.warning("ارسم مستطيل أولاً ثم اضغط حفظ.")

    else:
        st.info("ارفع Template أولاً.")

with colB:
    st.subheader("2) إعدادات المناطق المحفوظة")
    cfg: Optional[TemplateConfig] = st.session_state.cfg

    if cfg is None:
        st.info("بانتظار رفع Template.")
    else:
        # ID settings
        st.markdown("### ✅ كود الطالب (ID ROI)")
        st.write("ROI:", cfg.id_roi)
        cfg.id_digits = st.number_input("عدد خانات الكود", 1, 12, int(cfg.id_digits), 1)
        cfg.id_rows = st.number_input("عدد صفوف أرقام الكود (عادة 10)", 5, 12, int(cfg.id_rows), 1)

        st.markdown("---")
        st.markdown("### ✅ بلوكات الأسئلة Q Blocks")
        if not cfg.q_blocks:
            st.info("لم يتم إضافة أي Q Block بعد.")
        else:
            for i, b in enumerate(cfg.q_blocks, start=1):
                with st.expander(f"Block #{i}"):
                    b.start_q = st.number_input("Start Q", 1, 500, int(b.start_q), 1, key=f"sq_{i}")
                    b.end_q = st.number_input("End Q", 1, 500, int(b.end_q), 1, key=f"eq_{i}")
                    b.rows = st.number_input("Rows داخل البلوك", 1, 200, int(b.rows), 1, key=f"rw_{i}")
                    st.caption(f"ROI: x={b.x}, y={b.y}, w={b.w}, h={b.h}")

        st.markdown("---")
        st.markdown("### 💾 تصدير/استيراد Template JSON")
        js = config_to_json(cfg)
        st.download_button("⬇️ تحميل template.json", js.encode("utf-8"), "template.json", mime="application/json")

        upl = st.file_uploader("رفع template.json (اختياري)", type=["json"], key="json_upl")
        if upl:
            try:
                cfg2 = json_to_config(upl.getvalue().decode("utf-8"))
                st.session_state.cfg = cfg2
                st.success("✅ تم تحميل config")
                st.rerun()
            except Exception as e:
                st.error(f"JSON خطأ: {e}")

# -------------------------
# Step 3: Upload Roster + Key + Sheets
# -------------------------
st.markdown("---")
st.subheader("3) ملفات التصحيح")

c1, c2, c3 = st.columns([1, 1, 1], gap="large")
with c1:
    roster_file = st.file_uploader("📌 ملف الطلاب Roster (Excel/CSV) يحتوي: student_code, student_name", type=["xlsx", "xls", "csv"], key="roster")
with c2:
    key_file = st.file_uploader("✅ Answer Key (PDF صفحة واحدة أو صورة)", type=["pdf", "png", "jpg", "jpeg"], key="key")
with c3:
    sheets_file = st.file_uploader("🧾 أوراق الطلاب (PDF متعدد الصفحات أو صور)", type=["pdf", "png", "jpg", "jpeg"], key="sheets")

def load_roster(file) -> Dict[str, str]:
    if file.name.lower().endswith(("xlsx", "xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    if "student_code" not in df.columns or "student_name" not in df.columns:
        raise ValueError("Roster لازم يحتوي أعمدة: student_code و student_name")
    # treat codes as string (keep leading zeros if exists)
    df["student_code"] = df["student_code"].astype(str).str.strip()
    df["student_name"] = df["student_name"].astype(str).str.strip()
    return dict(zip(df["student_code"], df["student_name"]))

# -------------------------
# Run Grading
# -------------------------
st.markdown("---")
run_btn = st.button("🚀 ابدأ التصحيح", use_container_width=True)

if run_btn:
    if st.session_state.cfg is None or st.session_state.template_img is None:
        st.error("لازم ترفع Template وتحدد ID ROI + Q Blocks أولاً.")
        st.stop()

    if not (roster_file and key_file and sheets_file):
        st.error("ارفع Roster + Answer Key + أوراق الطلاب.")
        st.stop()

    cfg: TemplateConfig = st.session_state.cfg

    try:
        roster = load_roster(roster_file)
        st.success(f"✅ Roster: تم تحميل {len(roster)} طالب")
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Load Answer Key page
    key_pages = load_pages(key_file.getvalue(), key_file.name, dpi=dpi)
    key_img = key_pages[0].convert("RGB")
    key_thr = preprocess_for_omr(pil_to_bgr(key_img))
    key_ans = read_answers(key_thr, cfg, choices, min_fill=min_fill_ans, min_ratio=min_ratio)

    if not key_ans:
        st.error("لم يتم استخراج أي إجابات من Answer Key. تحقق من Q Blocks.")
        st.stop()

    # Load student pages
    pages = load_pages(sheets_file.getvalue(), sheets_file.name, dpi=dpi)

    results = []
    issues = []
    prog = st.progress(0)
    total = len(pages)

    for idx, pg in enumerate(pages, start=1):
        img = pg.convert("RGB")
        thr = preprocess_for_omr(pil_to_bgr(img))

        code, code_status = read_student_code(thr, cfg, min_fill=min_fill_id, min_ratio=min_ratio)
        # normalize code: keep as string, but don't kill leading zeros.
        code = code.strip()

        name = roster.get(code, "")

        stu_ans = read_answers(thr, cfg, choices, min_fill=min_fill_ans, min_ratio=min_ratio)

        # Score
        score = 0
        blank_cnt = 0
        double_cnt = 0

        for q, (ka, _) in key_ans.items():
            sa, stt = stu_ans.get(q, ("?", "MISSING"))
            if stt == "BLANK":
                blank_cnt += 1
            elif stt == "DOUBLE":
                double_cnt += 1

            # decide if question counted
            count_this = False
            if theory_ranges and in_ranges(q, theory_ranges):
                count_this = True
            if practical_ranges and in_ranges(q, practical_ranges):
                count_this = True

            # إذا المستخدم لم يحدد نطاقات: صحح كل الأسئلة الموجودة في key
            if not theory_ranges and not practical_ranges:
                count_this = True

            if not count_this:
                continue

            if sa == ka:
                score += 1
            else:
                # strict vs non-strict: كلاهما خطأ بالدرجة، لكن نسجل مشاكل فقط
                pass

        # Collect results
        results.append({
            "sheet_index": idx,
            "student_code": code,
            "student_name": name,
            "score": int(score),
        })

        # Collect issues (optional)
        if (code_status != "OK") or (blank_cnt > 0) or (double_cnt > 0) or (name == ""):
            issues.append({
                "sheet_index": idx,
                "student_code": code,
                "student_name": name,
                "code_status": code_status,
                "blank_count": blank_cnt,
                "double_count": double_cnt,
            })

        prog.progress(int(idx / total * 100))

    out_df = pd.DataFrame(results)
    issues_df = pd.DataFrame(issues)

    st.success("✅ تم التصحيح")

    # Download Excel
    buf = io.BytesIO()
    out_df.to_excel(buf, index=False)
    st.download_button("⬇️ تحميل النتائج Excel", buf.getvalue(), "results.xlsx")

    # Optional issues file
    if not issues_df.empty:
        buf2 = io.BytesIO()
        issues_df.to_excel(buf2, index=False)
        st.download_button("⬇️ تحميل ملف المشاكل (اختياري)", buf2.getvalue(), "issues.xlsx")

    st.dataframe(out_df, use_container_width=True)
