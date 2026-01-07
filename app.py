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
# Data models
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
    # إذا الاثنين فارغين = صحح الكل
    if not theory_ranges and not practical_ranges:
        return True
    return in_ranges(q, theory_ranges) or in_ranges(q, practical_ranges)


def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def rect_from_two_clicks(p1, p2, scale_back: float) -> Tuple[int, int, int, int]:
    # clicks are on resized image; convert back to original
    x1, y1 = int(p1["x"] / scale_back), int(p1["y"] / scale_back)
    x2, y2 = int(p2["x"] / scale_back), int(p2["y"] / scale_back)
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return x, y, w, h


# =========================
# OMR Reading
# =========================
def read_student_code(thr: np.ndarray, id_cfg: IdConfig) -> str:
    H, W = thr.shape[:2]
    x, y, w, h = clamp_roi(id_cfg.x, id_cfg.y, id_cfg.w, id_cfg.h, W, H)
    roi = thr[y:y + h, x:x + w]

    rows = id_cfg.rows
    cols = id_cfg.digits
    ch = max(1, h // rows)
    cw = max(1, w // cols)

    cell_area = max(1, ch * cw)
    min_fill = int(cell_area * 0.12)
    min_ratio = 1.25

    digits = []
    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            scores.append((str(r), score_cell(cell)))
        d, stt = pick_one(scores, min_fill, min_ratio)
        digits.append("" if d in ["?", "!"] else d)

    return "".join(digits)


def read_answers(thr: np.ndarray, qblock: QBlockConfig, choices: int) -> Dict[int, Tuple[str, str]]:
    H, W = thr.shape[:2]
    x, y, w, h = clamp_roi(qblock.x, qblock.y, qblock.w, qblock.h, W, H)
    roi = thr[y:y + h, x:x + w]

    letters = "ABCDE"[:choices]
    q_count = qblock.end_q - qblock.start_q + 1
    rows = qblock.rows if qblock.rows > 0 else q_count
    if abs(rows - q_count) > max(2, int(q_count * 0.25)):
        rows = q_count

    rh = max(1, h // rows)
    cw = max(1, w // choices)

    cell_area = max(1, rh * cw)
    min_fill = int(cell_area * 0.10)
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
        a, stt = pick_one(scores, min_fill, min_ratio)
        out[q] = (a, stt)
        q += 1

    return out


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="OMR (No Canvas) - Cloud Stable", layout="wide")
st.title("✅ OMR Bubble Sheet — حل نهائي بدون Canvas (يعمل على Streamlit Cloud)")

if "cfg" not in st.session_state:
    st.session_state.cfg = None
if "tpl_img" not in st.session_state:
    st.session_state.tpl_img = None
if "clicks" not in st.session_state:
    st.session_state.clicks = []  # list of click dicts on resized image


left, right = st.columns([1.15, 0.85], vertical_alignment="top")

with right:
    st.header("⚙️ الإعدادات")
    choices = st.radio("عدد الخيارات", [4, 5], horizontal=True)

    st.subheader("نطاق التصحيح")
    theory_txt = st.text_input("نطاق النظري (مثال 1-70 أو 1-40)", "")
    practical_txt = st.text_input("نطاق العملي (اختياري) (مثال 1-25)", "")
    theory_ranges = parse_ranges(theory_txt)
    practical_ranges = parse_ranges(practical_txt)

    st.divider()
    st.subheader("ملفات التصحيح")
    roster_file = st.file_uploader("Roster: student_code, student_name", type=["xlsx", "xls", "csv"])
    key_file = st.file_uploader("Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    sheets_file = st.file_uploader("Student Sheets (PDF/صور)", type=["pdf", "png", "jpg", "jpeg"])

    st.divider()
    st.subheader("Config JSON")
    cfg_up = st.file_uploader("رفع config.json", type=["json"])
    if cfg_up:
        try:
            cfg_obj = json.loads(cfg_up.getvalue().decode("utf-8"))
            st.session_state.cfg = TemplateConfig.from_dict(cfg_obj)
            st.success("✅ تم تحميل config.json")
        except Exception as e:
            st.error(f"فشل تحميل JSON: {e}")

    if st.session_state.cfg:
        cfg_bytes = json.dumps(st.session_state.cfg.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("⬇️ تنزيل config.json", cfg_bytes, file_name="config.json", mime="application/json")


with left:
    st.header("1) رفع نموذج الورقة ثم تحديد المناطق بالنقر (Click)")
    tpl_file = st.file_uploader("Template (PDF صفحة واحدة أو صورة)", type=["pdf", "png", "jpg", "jpeg"], key="tpl_upl")

    if tpl_file:
        pages = load_pages(tpl_file.getvalue(), tpl_file.name)
        tpl_img = pil_rgb(pages[0])
        st.session_state.tpl_img = tpl_img

        if st.session_state.cfg is None:
            st.session_state.cfg = TemplateConfig(page_w=tpl_img.size[0], page_h=tpl_img.size[1], id_cfg=None, q_blocks=[])
        else:
            # update page size if different
            st.session_state.cfg.page_w = tpl_img.size[0]
            st.session_state.cfg.page_h = tpl_img.size[1]

    if st.session_state.tpl_img is None:
        st.info("ارفع Template أولاً.")
    else:
        tpl_img = st.session_state.tpl_img
        page_w, page_h = tpl_img.size

        zoom_w = st.slider("عرض المعاينة (أكبر = أدق)", 800, 2200, 1400, 50)
        scale = zoom_w / page_w  # resized = original * scale
        resized_h = int(page_h * scale)
        preview = tpl_img.resize((zoom_w, resized_h))

        st.write("📌 **طريقة التحديد:** انقر مرتين: النقرة الأولى = أعلى-يسار، الثانية = أسفل-يمين.")
        mode = st.radio("ماذا تحدد الآن؟", ["ID (كود الطالب)", "Q Block (بلوك أسئلة)"], horizontal=True)

        # Click capture
        click = streamlit_image_coordinates(preview, key="img_click", width=zoom_w)

        if click:
            st.session_state.clicks.append(click)
            if len(st.session_state.clicks) > 2:
                # keep last two
                st.session_state.clicks = st.session_state.clicks[-2:]

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.write("آخر النقرات:")
            st.json(st.session_state.clicks)
        with c2:
            if st.button("🧹 مسح النقرات"):
                st.session_state.clicks = []
        with c3:
            if st.button("♻️ Reset Config"):
                st.session_state.cfg = TemplateConfig(page_w=page_w, page_h=page_h, id_cfg=None, q_blocks=[])
                st.session_state.clicks = []
                st.success("تم Reset.")

        if len(st.session_state.clicks) == 2:
            p1, p2 = st.session_state.clicks
            # scale back: click coords are on resized image => original = resized / scale
            x, y, w, h = rect_from_two_clicks(p1, p2, scale_back=scale)
            x, y, w, h = clamp_roi(x, y, w, h, page_w, page_h)

            st.success(f"✅ مستطيل جاهز: x={x}, y={y}, w={w}, h={h}")

            # show crop preview
            crop = tpl_img.crop((x, y, x + w, y + h))
            st.image(crop, caption="Preview Crop (من الصفحة الأصلية)", use_container_width=True)

            if mode.startswith("ID"):
                colA, colB, colC = st.columns([1, 1, 1])
                with colA:
                    digits = st.number_input("عدد خانات كود الطالب", 2, 12, 4, 1)
                with colB:
                    rows = st.number_input("عدد الصفوف (عادة 10)", 5, 15, 10, 1)
                with colC:
                    if st.button("✅ حفظ ID ROI"):
                        st.session_state.cfg.id_cfg = IdConfig(x=x, y=y, w=w, h=h, digits=int(digits), rows=int(rows))
                        st.success("تم حفظ ID ROI.")
                        st.session_state.clicks = []
            else:
                colA, colB, colC = st.columns([1, 1, 1])
                with colA:
                    start_q = st.number_input("Start Q", 1, 500, 1, 1)
                with colB:
                    end_q = st.number_input("End Q", 1, 500, 20, 1)
                with colC:
                    rows = st.number_input("Rows داخل البلوك", 1, 500, int(end_q - start_q + 1), 1)

                if st.button("✅ إضافة Q Block"):
                    qb = QBlockConfig(x=x, y=y, w=w, h=h, start_q=int(start_q), end_q=int(end_q), rows=int(rows))
                    st.session_state.cfg.q_blocks.append(qb)
                    st.success("تمت إضافة Q Block.")
                    st.session_state.clicks = []

        # config summary
        st.divider()
        st.subheader("ملخص الإعدادات")
        cfg = st.session_state.cfg
        if cfg.id_cfg:
            st.write("**ID ROI:**", cfg.id_cfg)
        else:
            st.warning("لم يتم تحديد ID ROI بعد.")
        st.write(f"**عدد Q Blocks:** {len(cfg.q_blocks)}")
        if cfg.q_blocks:
            st.dataframe(pd.DataFrame([asdict(b) for b in cfg.q_blocks]), use_container_width=True)


st.header("2) التصحيح وإخراج Excel")

if st.button("🚀 ابدأ التصحيح"):
    cfg: TemplateConfig = st.session_state.cfg
    if cfg is None:
        st.error("لا يوجد Config. ارفع Template وحدد المناطق.")
        st.stop()

    if cfg.id_cfg is None:
        st.error("حدد منطقة كود الطالب (ID ROI) أولاً.")
        st.stop()

    if not cfg.q_blocks:
        st.error("أضف Q Block واحد على الأقل للأسئلة.")
        st.stop()

    if not (roster_file and key_file and sheets_file):
        st.error("ارفع Roster + Answer Key + Student Sheets.")
        st.stop()

    # load roster
    if roster_file.name.lower().endswith((".xlsx", ".xls")):
        df_roster = pd.read_excel(roster_file)
    else:
        df_roster = pd.read_csv(roster_file)

    df_roster.columns = [c.strip().lower() for c in df_roster.columns]
    if "student_code" not in df_roster.columns or "student_name" not in df_roster.columns:
        st.error("Roster لازم يحتوي: student_code و student_name")
        st.stop()

    roster = dict(zip(df_roster["student_code"].astype(str), df_roster["student_name"].astype(str)))
    st.success(f"✅ Roster: {len(roster)} طالب")

    # load key
    key_pages = load_pages(key_file.getvalue(), key_file.name)
    key_img = pil_rgb(key_pages[0])
    key_thr = preprocess(pil_to_cv(key_img))

    key_answers: Dict[int, Tuple[str, str]] = {}
    for qb in cfg.q_blocks:
        key_answers.update(read_answers(key_thr, qb, choices))

    # load student sheets
    pages = load_pages(sheets_file.getvalue(), sheets_file.name)
    total_pages = len(pages)

    results = []
    prog = st.progress(0)

    for idx, pg in enumerate(pages, 1):
        img = pil_rgb(pg)
        thr = preprocess(pil_to_cv(img))

        code = read_student_code(thr, cfg.id_cfg)
        name = roster.get(str(code), "")

        stu_answers: Dict[int, Tuple[str, str]] = {}
        for qb in cfg.q_blocks:
            stu_answers.update(read_answers(thr, qb, choices))

        score = 0
        total_counted = 0

        for q, (ka, _) in key_answers.items():
            if not should_count_question(q, theory_ranges, practical_ranges):
                continue
            total_counted += 1
            sa, _ = stu_answers.get(q, ("?", "MISSING"))
            if sa == ka:
                score += 1

        results.append({
            "sheet_index": idx,
            "student_code": code,
            "student_name": name,
            "score": int(score),
            "total_questions": int(total_counted),
        })

        prog.progress(int(idx / total_pages * 100))

    out_df = pd.DataFrame(results)
    st.success("✅ تم التصحيح")
    st.dataframe(out_df.head(50), use_container_width=True)

    buf = io.BytesIO()
    out_df.to_excel(buf, index=False)
    st.download_button(
        "⬇️ تحميل results.xlsx",
        buf.getvalue(),
        file_name="results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
