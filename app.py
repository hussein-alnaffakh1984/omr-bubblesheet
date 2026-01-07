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
    rows: int  # number of question rows inside this block


@dataclass
class TemplateConfig:
    # Image reference size (the template page size used during drawing)
    ref_w: int = 0
    ref_h: int = 0

    # Student ID ROI (bubble digits area)
    id_roi: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h
    id_digits: int = 4   # number of digits columns
    id_rows: int = 10    # digit rows (0-9)

    # Question blocks
    q_blocks: List[QBlock] = None

    def to_dict(self):
        d = asdict(self)
        d["q_blocks"] = [asdict(b) for b in (self.q_blocks or [])]
        return d

    @staticmethod
    def from_dict(d: dict):
        cfg = TemplateConfig()
        cfg.ref_w = int(d.get("ref_w", 0))
        cfg.ref_h = int(d.get("ref_h", 0))
        cfg.id_roi = tuple(d["id_roi"]) if d.get("id_roi") else None
        cfg.id_digits = int(d.get("id_digits", 4))
        cfg.id_rows = int(d.get("id_rows", 10))
        cfg.q_blocks = [QBlock(**qb) for qb in d.get("q_blocks", [])]
        return cfg


# =========================
# Helpers: Parsing ranges
# =========================
def parse_ranges(txt: str) -> List[Tuple[int, int]]:
    """
    "1-40, 45, 50-60" -> [(1,40),(45,45),(50,60)]
    """
    if not txt or not txt.strip():
        return []
    out = []
    for part in txt.split(","):
        p = part.strip()
        if not p:
            continue
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
# Image I/O & Preprocess
# =========================
def load_pages(file_bytes: bytes, filename: str) -> List[Image.Image]:
    name = filename.lower()
    if name.endswith(".pdf"):
        # DPI moderate (balances quality & speed on cloud)
        return convert_from_bytes(file_bytes, dpi=200)
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess_to_binary(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )
    return thr


def count_ink(bin_roi: np.ndarray) -> int:
    return int(np.sum(bin_roi > 0))


def pick_one(scores: List[Tuple[str, int]], min_fill: int, min_ratio: float) -> Tuple[str, str]:
    """
    Returns (choice, status)
    status: OK, BLANK, DOUBLE
    choice: 'A'.. or '?' or '!'
    """
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0

    if top_s < min_fill:
        return "?", "BLANK"

    # double if top and second are close
    if second_s > 0 and (top_s / (second_s + 1e-6)) < min_ratio:
        return "!", "DOUBLE"

    return top_c, "OK"


# =========================
# Coordinate mapping (canvas->original)
# =========================
def canvas_to_original_rect(
    rect_canvas: Tuple[int, int, int, int],
    canvas_w: int,
    canvas_h: int,
    orig_w: int,
    orig_h: int
) -> Tuple[int, int, int, int]:
    x, y, w, h = rect_canvas
    sx = orig_w / float(canvas_w)
    sy = orig_h / float(canvas_h)
    ox = int(round(x * sx))
    oy = int(round(y * sy))
    ow = int(round(w * sx))
    oh = int(round(h * sy))
    return ox, oy, ow, oh


def rect_from_canvas_obj(obj: dict) -> Optional[Tuple[int, int, int, int]]:
    # Fabric.js rect properties
    if obj.get("type") != "rect":
        return None
    left = obj.get("left", 0)
    top = obj.get("top", 0)
    width = obj.get("width", 0)
    height = obj.get("height", 0)
    scaleX = obj.get("scaleX", 1.0)
    scaleY = obj.get("scaleY", 1.0)
    x = int(round(left))
    y = int(round(top))
    w = int(round(width * scaleX))
    h = int(round(height * scaleY))
    # sanity
    if w <= 2 or h <= 2:
        return None
    return (x, y, w, h)


# =========================
# OMR Reading using TemplateConfig
# =========================
def read_student_code(thr: np.ndarray, cfg: TemplateConfig, min_fill=200, min_ratio=1.25) -> Tuple[str, Dict]:
    """
    Reads numeric bubble code from cfg.id_roi as a grid:
      rows = cfg.id_rows (0..9)
      cols = cfg.id_digits
    """
    debug = {"status": "NO_ID_ROI", "digits": [], "raw": []}
    if cfg.id_roi is None:
        return "", debug

    x, y, w, h = cfg.id_roi
    roi = thr[y:y+h, x:x+w]
    rows = cfg.id_rows
    cols = cfg.id_digits

    ch = max(1, h // rows)
    cw = max(1, w // cols)

    digits = []
    for c in range(cols):
        scores = []
        raw_col = []
        for r in range(rows):
            cell = roi[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            s = count_ink(cell)
            scores.append((str(r), s))
            raw_col.append(s)

        d, stt = pick_one(scores, min_fill=min_fill, min_ratio=min_ratio)
        digits.append("" if d in ["?", "!"] else d)

        debug["raw"].append(raw_col)
        debug["digits"].append({"col": c, "picked": d, "status": stt, "scores": scores})

    code = "".join(digits)
    debug["status"] = "OK" if code else "EMPTY"
    return code, debug


def read_answers(thr: np.ndarray, cfg: TemplateConfig, choices: int, min_fill=160, min_ratio=1.25) -> Tuple[Dict[int, Tuple[str, str]], Dict]:
    """
    Reads answers for all cfg.q_blocks.
    Each block: rows (question rows), choices columns.
    """
    letters = "ABCDE"[:choices]
    out: Dict[int, Tuple[str, str]] = {}
    debug = {"blocks": []}

    if not cfg.q_blocks:
        return out, {"blocks": [], "status": "NO_Q_BLOCKS"}

    for bi, b in enumerate(cfg.q_blocks, 1):
        x, y, w, h = b.x, b.y, b.w, b.h
        roi = thr[y:y+h, x:x+w]
        rows = max(1, b.rows)
        rh = max(1, h // rows)
        cw = max(1, w // choices)

        q = b.start_q
        block_dbg = {"block_index": bi, "start_q": b.start_q, "end_q": b.end_q, "rows": rows, "items": []}

        for r in range(rows):
            if q > b.end_q:
                break

            scores = []
            for c in range(choices):
                cell = roi[r*rh:(r+1)*rh, c*cw:(c+1)*cw]
                s = count_ink(cell)
                scores.append((letters[c], s))

            a, stt = pick_one(scores, min_fill=min_fill, min_ratio=min_ratio)
            out[q] = (a, stt)
            block_dbg["items"].append({"q": q, "picked": a, "status": stt, "scores": scores})
            q += 1

        debug["blocks"].append(block_dbg)

    debug["status"] = "OK"
    return out, debug


# =========================
# UI: Config persistence
# =========================
def ensure_state():
    if "cfg" not in st.session_state:
        st.session_state.cfg = TemplateConfig(q_blocks=[])
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = "canvas_v1"
    if "last_rect" not in st.session_state:
        st.session_state.last_rect = None
    if "template_page" not in st.session_state:
        st.session_state.template_page = None  # PIL
    if "template_orig_size" not in st.session_state:
        st.session_state.template_orig_size = (0, 0)


def reset_canvas():
    st.session_state.canvas_key = st.session_state.canvas_key + "_r"
    st.session_state.last_rect = None


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="OMR Bubble Sheet (Remark-Style)", layout="wide")
ensure_state()

st.title("✅ برنامج تصحيح ببل شيت (Remark-Style) — Cloud")
st.caption("ترسم مناطق الكود والأسئلة بالماوس ثم يصحح ويصدر Excel: sheet_index, student_code, student_name, score.")

left, right = st.columns([1.25, 0.75], gap="large")

with right:
    st.subheader("⚙️ إعدادات عامة")

    choices = st.radio("عدد الخيارات", [4, 5], horizontal=True)
    strict_mode = st.checkbox("وضع صارم: BLANK/DOUBLE = خطأ", value=True)

    st.markdown("---")
    st.subheader("📌 نطاق التصحيح")
    theory_txt = st.text_input("نطاق النظري (مثال: 1-40 أو 1-70,80-95)", value="")
    practical_txt = st.text_input("نطاق العملي (اختياري)", value="")

    theory_ranges = parse_ranges(theory_txt)
    practical_ranges = parse_ranges(practical_txt)

    st.markdown("---")
    st.subheader("🆔 إعداد كود الطالب")
    st.session_state.cfg.id_digits = st.number_input("عدد خانات الكود", min_value=1, max_value=12, value=st.session_state.cfg.id_digits, step=1)
    st.session_state.cfg.id_rows = st.number_input("عدد صفوف أرقام الكود (عادة 10)", min_value=5, max_value=20, value=st.session_state.cfg.id_rows, step=1)

    st.markdown("---")
    st.subheader("🧱 إعداد بلوك الأسئلة قبل الحفظ")
    tmp_start_q = st.number_input("Start Q", min_value=1, value=1, step=1)
    tmp_end_q = st.number_input("End Q", min_value=1, value=20, step=1)
    tmp_rows = st.number_input("Rows داخل البلوك", min_value=1, value=20, step=1)

    st.markdown("---")
    st.subheader("🧩 إدارة القالب (Template)")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Clear Canvas", use_container_width=True):
            reset_canvas()
    with c2:
        if st.button("♻️ Reset Template", use_container_width=True):
            st.session_state.cfg = TemplateConfig(q_blocks=[])
            st.session_state.last_rect = None
            reset_canvas()

    # Export / Import template json
    st.markdown("**حفظ/تحميل قالب JSON**")
    cfg_json = json.dumps(st.session_state.cfg.to_dict(), ensure_ascii=False, indent=2)
    st.download_button("⬇️ Download template.json", data=cfg_json.encode("utf-8"), file_name="template.json", mime="application/json")

    up_cfg = st.file_uploader("⬆️ Upload template.json", type=["json"], key="cfg_up")
    if up_cfg:
        try:
            d = json.loads(up_cfg.getvalue().decode("utf-8"))
            st.session_state.cfg = TemplateConfig.from_dict(d)
            st.success("تم تحميل القالب ✅")
        except Exception as e:
            st.error(f"فشل تحميل القالب: {e}")


with left:
    st.subheader("1) ارفع ورقة نموذج (Template) ثم ارسم عليها")
    template_file = st.file_uploader("Template: PDF صفحة واحدة أو صورة", type=["pdf", "png", "jpg", "jpeg"])

    if template_file:
        pages = load_pages(template_file.getvalue(), template_file.name)
        template_img = pages[0].convert("RGB")
        st.session_state.template_page = template_img
        tw, th = template_img.size
        st.session_state.template_orig_size = (tw, th)
        st.session_state.cfg.ref_w = tw
        st.session_state.cfg.ref_h = th

        # Canvas sizing
        canvas_w = st.slider("Canvas width", 700, 1600, min(1100, tw), 10)
        canvas_h = int(canvas_w * (th / tw))
        preview = template_img.resize((canvas_w, canvas_h))

        st.info("✍️ اختر ماذا ترسم: ID ROI أو Q Block ثم اسحب بالماوس لرسم مستطيل فوق الورقة مباشرة.")

        draw_mode = st.radio("ماذا تريد ترسم الآن؟", ["ID ROI (كود الطالب)", "Q Block (بلوك أسئلة)"], horizontal=True)

        # Canvas with background image (this fixes the 'white canvas' and drawing issues)
        canvas_result = st_canvas(
            background_image=preview,
            drawing_mode="rect",
            stroke_width=3,
            stroke_color="#ff0000",
            fill_color="rgba(255, 0, 0, 0.20)",
            height=canvas_h,
            width=canvas_w,
            key=st.session_state.canvas_key,
        )

        # Capture last rectangle
        if canvas_result and canvas_result.json_data:
            objs = canvas_result.json_data.get("objects", [])
            if objs:
                rc = rect_from_canvas_obj(objs[-1])
                if rc:
                    st.session_state.last_rect = rc

        # Show last rect and provide save buttons
        if st.session_state.last_rect:
            x, y, w, h = st.session_state.last_rect
            st.success(f"آخر مستطيل: x={x}, y={y}, w={w}, h={h}")

            # Map to original coordinates
            ox, oy, ow, oh = canvas_to_original_rect(
                st.session_state.last_rect,
                canvas_w, canvas_h,
                tw, th
            )
            st.caption(f"إحداثيات على الصورة الأصلية: x={ox}, y={oy}, w={ow}, h={oh}")

            b1, b2, b3 = st.columns([1, 1, 1])
            with b1:
                if st.button("💾 Save ID ROI", use_container_width=True):
                    st.session_state.cfg.id_roi = (ox, oy, ow, oh)
                    st.success("تم حفظ ID ROI ✅")
                    reset_canvas()

            with b2:
                if st.button("➕ Add Q Block", use_container_width=True):
                    qb = QBlock(
                        x=ox, y=oy, w=ow, h=oh,
                        start_q=int(tmp_start_q),
                        end_q=int(tmp_end_q),
                        rows=int(tmp_rows),
                    )
                    if st.session_state.cfg.q_blocks is None:
                        st.session_state.cfg.q_blocks = []
                    st.session_state.cfg.q_blocks.append(qb)
                    st.success("تمت إضافة Q Block ✅")
                    reset_canvas()

            with b3:
                if st.button("🗑 Delete Last Block", use_container_width=True):
                    if st.session_state.cfg.q_blocks:
                        st.session_state.cfg.q_blocks.pop()
                        st.warning("تم حذف آخر Q Block")
                    reset_canvas()

        # Display saved config summary
        st.markdown("---")
        st.subheader("📌 الملخص الحالي للقالب")
        cfg = st.session_state.cfg
        st.write("**ID ROI:**", cfg.id_roi)
        st.write("**Q Blocks:**", len(cfg.q_blocks or []))
        if cfg.q_blocks:
            for i, b in enumerate(cfg.q_blocks, 1):
                st.write(f"Block {i}: (x={b.x}, y={b.y}, w={b.w}, h={b.h}) | Q {b.start_q}-{b.end_q} | rows={b.rows}")

    else:
        st.warning("ارفع Template أولاً حتى تستطيع الرسم وتحديد المناطق.")


st.markdown("---")
st.subheader("2) Roster (أسماء الطلاب)")
st.caption("ارفع ملف Excel/CSV يحتوي عمودين: student_code , student_name")

# Download roster template
roster_template = pd.DataFrame({"student_code": ["1234", "5678"], "student_name": ["Student A", "Student B"]})
buf_roster = io.BytesIO()
roster_template.to_excel(buf_roster, index=False)
st.download_button("⬇️ Download roster template", data=buf_roster.getvalue(), file_name="roster_template.xlsx")

roster_file = st.file_uploader("Roster file", type=["xlsx", "xls", "csv"], key="roster_up")
roster_map: Dict[str, str] = {}
if roster_file:
    if roster_file.name.lower().endswith(".csv"):
        df = pd.read_csv(roster_file)
    else:
        df = pd.read_excel(roster_file)

    df.columns = [c.strip().lower() for c in df.columns]
    if "student_code" not in df.columns or "student_name" not in df.columns:
        st.error("يجب أن يحتوي الملف على عمودين: student_code و student_name")
    else:
        df["student_code"] = df["student_code"].astype(str).str.strip()
        df["student_name"] = df["student_name"].astype(str).str.strip()
        roster_map = dict(zip(df["student_code"], df["student_name"]))
        st.success(f"تم تحميل roster: {len(roster_map)} طالب ✅")


st.markdown("---")
st.subheader("3) Answer Key + Student Sheets")
key_file = st.file_uploader("Answer Key (PDF صفحة واحدة أو صورة)", type=["pdf", "png", "jpg", "jpeg"], key="key_up")
sheets_file = st.file_uploader("Student Sheets (PDF متعدد الصفحات أو صور)", type=["pdf", "png", "jpg", "jpeg"], key="sheets_up")

st.markdown("---")
colA, colB = st.columns([1, 1])
with colA:
    show_debug = st.checkbox("عرض Debug (لأول 3 أوراق)", value=False)
with colB:
    run_btn = st.button("🚀 ابدأ التصحيح", use_container_width=True)

if run_btn:
    cfg = st.session_state.cfg
    if st.session_state.template_page is None:
        st.error("ارفع Template وحدد ID ROI و Q Blocks أولاً.")
        st.stop()

    if cfg.id_roi is None:
        st.error("لم يتم تحديد ID ROI (منطقة كود الطالب).")
        st.stop()

    if not cfg.q_blocks:
        st.error("لم يتم إضافة أي Q Block (بلوك أسئلة).")
        st.stop()

    if not roster_map:
        st.error("ارفع ملف roster أولاً (student_code, student_name).")
        st.stop()

    if not key_file or not sheets_file:
        st.error("ارفع Answer Key و Student Sheets.")
        st.stop()

    # Load key page and student pages
    key_pages = load_pages(key_file.getvalue(), key_file.name)
    key_img = key_pages[0].convert("RGB")
    key_thr = preprocess_to_binary(pil_to_bgr(key_img))
    key_ans, key_dbg = read_answers(key_thr, cfg, choices)

    stu_pages = load_pages(sheets_file.getvalue(), sheets_file.name)

    results = []
    issues = []

    prog = st.progress(0)
    for i, pg in enumerate(stu_pages, 1):
        thr = preprocess_to_binary(pil_to_bgr(pg))
        code, code_dbg = read_student_code(thr, cfg)
        name = roster_map.get(code, "")

        stu_ans, stu_dbg = read_answers(thr, cfg, choices)

        score = 0
        total = 0

        for q, (ka, kst) in key_ans.items():
            # Decide whether this question is included in scoring
            included = False
            if theory_ranges and in_ranges(q, theory_ranges):
                included = True
            if practical_ranges and in_ranges(q, practical_ranges):
                included = True

            # If user did not specify ranges, score all questions in key
            if not theory_ranges and not practical_ranges:
                included = True

            if not included:
                continue

            total += 1
            sa, sst = stu_ans.get(q, ("?", "MISSING"))

            # Strict mode: BLANK/DOUBLE/MISSING treated wrong
            if strict_mode:
                if sa == ka and sst == "OK":
                    score += 1
                else:
                    if sst in ("BLANK", "DOUBLE", "MISSING"):
                        issues.append({
                            "sheet_index": i,
                            "student_code": code,
                            "student_name": name,
                            "q": q,
                            "status": sst,
                            "student_mark": sa,
                            "key": ka
                        })
            else:
                # Non-strict: only compare letter if not blank/double
                if sa == ka:
                    score += 1
                if sst in ("BLANK", "DOUBLE", "MISSING"):
                    issues.append({
                        "sheet_index": i,
                        "student_code": code,
                        "student_name": name,
                        "q": q,
                        "status": sst,
                        "student_mark": sa,
                        "key": ka
                    })

        results.append({
            "sheet_index": i,
            "student_code": code,
            "student_name": name,
            "score": score,
            "total": total
        })

        if show_debug and i <= 3:
            st.write(f"--- Debug page {i} ---")
            st.write("Code:", code, "Name:", name)
            st.json({"code_debug": code_dbg, "answers_debug_sample": stu_dbg["blocks"][0]["items"][:3] if stu_dbg.get("blocks") else []})

        prog.progress(int(i / len(stu_pages) * 100))

    df_out = pd.DataFrame(results)
    df_issues = pd.DataFrame(issues)

    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="results")
        if len(df_issues) > 0:
            df_issues.to_excel(writer, index=False, sheet_name="issues")

    st.success("✅ تم التصحيح وإنشاء ملف Excel")
    st.download_button("⬇️ تحميل النتائج Excel", data=out_buf.getvalue(), file_name="results.xlsx")
    st.dataframe(df_out, use_container_width=True)
    if len(df_issues) > 0:
        st.warning(f"يوجد {len(df_issues)} مشكلة (Blank/Double/Missing) — راجع Sheet: issues")
        st.dataframe(df_issues.head(50), use_container_width=True)
