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
    id_roi: Tuple[int, int, int, int] = (0, 0, 0, 0)
    id_digits: int = 4
    id_rows: int = 10
    q_blocks: List[QBlock] = None
    choices: int = 4

    def to_jsonable(self):
        d = asdict(self)
        d["q_blocks"] = [asdict(b) for b in (self.q_blocks or [])]
        return d


# =========================
# Image Helpers
# =========================
def load_pages(file_bytes: bytes, filename: str) -> List[Image.Image]:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        try:
            pages = convert_from_bytes(file_bytes, dpi=200, fmt="png")  # خفضنا DPI لسرعة أعلى
            return pages
        except Exception as e:
            st.error(f"خطأ في قراءة PDF: {e}")
            return []
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return [img]
    except Exception as e:
        st.error(f"خطأ في قراءة الصورة: {e}")
        return []


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_to(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)


# =========================
# Alignment - مبسط وأسرع
# =========================
def simple_align(bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    """محاذاة بسيطة وسريعة"""
    h, w = bgr.shape[:2]
    
    # تصحيح انحراف بسيط فقط
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is not None and len(lines) > 0:
        angles = []
        for rho, theta in lines[:10]:
            angle = (theta - np.pi/2) * 180 / np.pi
            if abs(angle) < 45:
                angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                bgr = cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return resize_to(bgr, tw, th)


# =========================
# Bubble Processing - محسّن للسرعة
# =========================
def preprocess_fast(bgr: np.ndarray) -> np.ndarray:
    """معالجة سريعة"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # kernel أصغر
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 6)  # معايير أسرع
    return thr


def score_cell(bin_cell: np.ndarray) -> float:
    """حساب سريع للتظليل"""
    if bin_cell.size == 0:
        return 0.0
    
    h, w = bin_cell.shape[:2]
    mx = int(w * 0.25)
    my = int(h * 0.25)
    
    if h - 2*my <= 0 or w - 2*mx <= 0:
        return 0.0
    
    c = bin_cell[my:h-my, mx:w-mx]
    return float(np.sum(c > 0)) / float(c.size)


def pick_one(scores: List[Tuple[str, float]], min_fill=0.20):
    """اختيار مبسط"""
    if not scores:
        return "?", "ERROR", 0.0, 0.0
    
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0.0

    if top_s < min_fill:
        return "?", "BLANK", top_s, second_s
    if second_s > min_fill and (top_s / (second_s + 1e-9)) < 1.4:
        return "!", "DOUBLE", top_s, second_s
    return top_c, "OK", top_s, second_s


# =========================
# Read Functions
# =========================
def read_student_code(thr: np.ndarray, cfg: TemplateConfig) -> Tuple[str, Dict]:
    x, y, w, h = cfg.id_roi
    if w <= 0 or h <= 0:
        return "", {"error": "ID ROI not set"}

    img_h, img_w = thr.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return "", {"error": "ID ROI out of bounds"}

    roi = thr[y:y + h, x:x + w]
    rows = cfg.id_rows
    cols = cfg.id_digits
    ch = h // rows
    cw = w // cols

    digits = []
    for c in range(cols):
        col_scores = []
        for r in range(rows):
            cell = roi[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            fill = score_cell(cell)
            col_scores.append((str(r), fill))
        
        d, _, _, _ = pick_one(col_scores, min_fill=0.18)
        digits.append("" if d in ("?", "!") else d)

    return "".join(digits), {}


def read_answers(thr: np.ndarray, block: QBlock, choices: int) -> Dict[int, Tuple[str, str, float, float]]:
    letters = "ABCDEFGH"[:choices]
    out = {}

    x, y, w, h = block.x, block.y, block.w, block.h
    img_h, img_w = thr.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return out

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
        
        a, status, top, second = pick_one(scores, min_fill=0.20)
        out[q] = (a, status, top, second)
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
# Draw Preview - مبسط
# =========================
def draw_preview(img: Image.Image, cfg: TemplateConfig) -> Image.Image:
    im = img.copy().convert("RGB")
    dr = ImageDraw.Draw(im)

    x, y, w, h = cfg.id_roi
    if w > 0 and h > 0:
        dr.rectangle([x, y, x + w, y + h], outline="red", width=3)
        dr.text((x + 5, y + 5), "ID", fill="red")

    for i, b in enumerate(cfg.q_blocks or [], 1):
        dr.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline="green", width=3)
        dr.text((b.x + 5, b.y + 5), f"Q{i}", fill="green")
    
    return im


# =========================
# UI - مبسط وسريع
# =========================
st.set_page_config(page_title="OMR Fast", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    .stButton>button {width: 100%; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ OMR Bubble Sheet - نسخة سريعة")

# Session
if "cfg" not in st.session_state:
    st.session_state.cfg = TemplateConfig(q_blocks=[])
if "template_img" not in st.session_state:
    st.session_state.template_img = None
if "rect_start" not in st.session_state:
    st.session_state.rect_start = None

# Layout
col1, col2 = st.columns([2, 1])

# =========================
# RIGHT PANEL
# =========================
with col2:
    st.subheader("⚙️ الإعدادات")
    
    tpl = st.file_uploader("📄 النموذج", type=["pdf", "png", "jpg"])
    
    if tpl:
        with st.spinner("⏳ جاري التحميل..."):
            pages = load_pages(tpl.getvalue(), tpl.name)
            if pages:
                st.session_state.template_img = pages[0].convert("RGB")
                tw, th = st.session_state.template_img.size
                st.session_state.cfg.template_w = tw
                st.session_state.cfg.template_h = th
                st.success(f"✅ {tw}x{th}")
    
    st.divider()
    
    choices = st.selectbox("عدد الخيارات", [4, 5, 6], 0)
    st.session_state.cfg.choices = choices
    
    col_a, col_b = st.columns(2)
    with col_a:
        id_digits = st.number_input("خانات الكود", 1, 12, 4, 1)
    with col_b:
        id_rows = st.number_input("صفوف الكود", 5, 15, 10, 1)
    
    st.session_state.cfg.id_digits = id_digits
    st.session_state.cfg.id_rows = id_rows
    
    st.divider()
    
    mode = st.radio("التحديد", ["🆔 ID", "📝 أسئلة"], 0)
    
    if mode == "📝 أسئلة":
        col_c, col_d, col_e = st.columns(3)
        with col_c:
            b_start = st.number_input("من", 1, 500, 1, 1)
        with col_d:
            b_end = st.number_input("إلى", 1, 500, 20, 1)
        with col_e:
            b_rows = st.number_input("صفوف", 1, 200, 20, 1)
    else:
        b_start = b_end = b_rows = 0
    
    st.info("💡 اضغط مرتين: الزاوية الأولى ثم الثانية")
    
    if st.button("🗑️ مسح التحديد"):
        st.session_state.rect_start = None
        st.success("✅ تم المسح")
    
    if st.button("🔄 Reset الكل"):
        st.session_state.cfg.id_roi = (0, 0, 0, 0)
        st.session_state.cfg.q_blocks = []
        st.session_state.rect_start = None
        st.success("✅ تم Reset")
    
    st.divider()
    
    roster_file = st.file_uploader("📋 Roster", type=["xlsx", "csv"])
    key_file = st.file_uploader("🔑 Answer Key", type=["pdf", "png", "jpg"])
    sheets_file = st.file_uploader("📚 أوراق الطلاب", type=["pdf", "png", "jpg"])
    
    theory_txt = st.text_input("النظري", "", placeholder="1-40")
    practical_txt = st.text_input("العملي", "", placeholder="41-60")
    strict = st.checkbox("وضع صارم", True)

# =========================
# LEFT PANEL
# =========================
with col1:
    if st.session_state.template_img is None:
        st.info("📄 ارفع النموذج من اليمين")
        st.stop()
    
    st.subheader("🖱️ اضغط مرتين لتحديد المستطيل")
    
    # رسم الصورة مع المستطيلات
    preview = draw_preview(st.session_state.template_img, st.session_state.cfg)
    
    # عرض الصورة مع إمكانية النقر (طريقة مبسطة)
    st.image(preview, use_column_width=True)
    
    st.info("⚠️ استخدم الإدخال اليدوي للإحداثيات أدناه (أسرع)")
    
    # إدخال يدوي مبسط
    st.subheader("📍 إدخال يدوي للإحداثيات")
    
    col_x1, col_y1, col_x2, col_y2 = st.columns(4)
    with col_x1:
        x1 = st.number_input("X1", 0, st.session_state.cfg.template_w, 0, key="x1")
    with col_y1:
        y1 = st.number_input("Y1", 0, st.session_state.cfg.template_h, 0, key="y1")
    with col_x2:
        x2 = st.number_input("X2", 0, st.session_state.cfg.template_w, 100, key="x2")
    with col_y2:
        y2 = st.number_input("Y2", 0, st.session_state.cfg.template_h, 100, key="y2")
    
    if st.button("💾 حفظ المستطيل", type="primary"):
        x = int(min(x1, x2))
        y = int(min(y1, y2))
        w = int(abs(x2 - x1))
        h = int(abs(y2 - y1))
        
        if w < 10 or h < 10:
            st.error("❌ المستطيل صغير جداً")
        else:
            if mode == "🆔 ID":
                st.session_state.cfg.id_roi = (x, y, w, h)
                st.success(f"✅ تم حفظ ID ROI: ({x}, {y}, {w}, {h})")
            else:
                qb = QBlock(x=x, y=y, w=w, h=h,
                          start_q=int(b_start), end_q=int(b_end), rows=int(b_rows))
                st.session_state.cfg.q_blocks.append(qb)
                st.success(f"✅ تم إضافة Q Block: {b_start}-{b_end}")
    
    if st.session_state.cfg.q_blocks:
        st.write("**البلوكات:**")
        for i, b in enumerate(st.session_state.cfg.q_blocks, 1):
            col_info, col_del = st.columns([3, 1])
            with col_info:
                st.text(f"{i}. Q{b.start_q}-{b.end_q} ({b.rows}r)")
            with col_del:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.cfg.q_blocks.pop(i-1)
                    st.rerun()
    
    st.divider()
    
    # =========================
    # GRADING
    # =========================
    if st.button("🚀 ابدأ التصحيح", type="primary"):
        cfg = st.session_state.cfg
        
        if cfg.id_roi[2] <= 0:
            st.error("❌ حدد ID ROI")
            st.stop()
        if not cfg.q_blocks:
            st.error("❌ أضف بلوك أسئلة")
            st.stop()
        if not (roster_file and key_file and sheets_file):
            st.error("❌ ارفع جميع الملفات")
            st.stop()
        
        try:
            # Roster
            if roster_file.name.endswith(("xlsx", "xls")):
                df_roster = pd.read_excel(roster_file)
            else:
                df_roster = pd.read_csv(roster_file)
            
            df_roster.columns = [c.strip().lower().replace(" ", "_") for c in df_roster.columns]
            roster = dict(zip(df_roster["student_code"].astype(str).str.strip(),
                             df_roster["student_name"].astype(str).str.strip()))
            
            # Answer Key
            key_pages = load_pages(key_file.getvalue(), key_file.name)
            key_bgr = pil_to_bgr(key_pages[0])
            key_bgr = simple_align(key_bgr, cfg.template_w, cfg.template_h)
            key_thr = preprocess_fast(key_bgr)
            
            key_ans = {}
            for b in cfg.q_blocks:
                for q, (ans, _, _, _) in read_answers(key_thr, b, cfg.choices).items():
                    key_ans[q] = ans
            
            st.success(f"✅ {len(key_ans)} سؤال في Answer Key")
            
            # Ranges
            theory_ranges = parse_ranges(theory_txt)
            practical_ranges = parse_ranges(practical_txt)
            
            # Student sheets
            pages = load_pages(sheets_file.getvalue(), sheets_file.name)
            st.success(f"✅ {len(pages)} ورقة")
            
            prog = st.progress(0)
            results = []
            
            for idx, pg in enumerate(pages, 1):
                prog.progress(idx / len(pages))
                
                bgr = pil_to_bgr(pg)
                bgr = simple_align(bgr, cfg.template_w, cfg.template_h)
                thr = preprocess_fast(bgr)
                
                code, _ = read_student_code(thr, cfg)
                code = code.strip() or f"UNKNOWN_{idx}"
                name = roster.get(code, "غير موجود")
                
                stu_ans = {}
                for b in cfg.q_blocks:
                    stu_ans.update(read_answers(thr, b, cfg.choices))
                
                score_t = score_p = total_t = total_p = 0
                
                for q, ka in key_ans.items():
                    in_t = theory_ranges and in_ranges(q, theory_ranges)
                    in_p = practical_ranges and in_ranges(q, practical_ranges)
                    
                    if not theory_ranges and not practical_ranges:
                        in_t = True
                    
                    if not (in_t or in_p):
                        continue
                    
                    sa, stt, _, _ = stu_ans.get(q, ("?", "MISSING", 0, 0))
                    
                    if strict and stt in ("BLANK", "DOUBLE"):
                        is_correct = False
                    else:
                        is_correct = (sa == ka)
                    
                    if in_t:
                        total_t += 1
                        if is_correct:
                            score_t += 1
                    if in_p:
                        total_p += 1
                        if is_correct:
                            score_p += 1
                
                pct_t = (score_t / total_t * 100) if total_t > 0 else 0
                pct_p = (score_p / total_p * 100) if total_p > 0 else 0
                total = score_t + score_p
                total_q = total_t + total_p
                pct_total = (total / total_q * 100) if total_q > 0 else 0
                
                results.append({
                    "الورقة": idx,
                    "الكود": code,
                    "الاسم": name,
                    "النظري": f"{score_t}/{total_t}",
                    "نسبة_نظري": f"{pct_t:.1f}%",
                    "العملي": f"{score_p}/{total_p}",
                    "نسبة_عملي": f"{pct_p:.1f}%",
                    "الإجمالي": f"{total}/{total_q}",
                    "النسبة": f"{pct_total:.1f}%"
                })
            
            prog.empty()
            
            df = pd.DataFrame(results)
            st.success("✅ اكتمل التصحيح!")
            st.dataframe(df, use_container_width=True)
            
            # Stats
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("الأوراق", len(df))
            with col_s2:
                avg = df["النسبة"].str.rstrip('%').astype(float).mean()
                st.metric("المتوسط", f"{avg:.1f}%")
            
            # Excel
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            
            st.download_button("⬇️ تحميل Excel", buf.getvalue(),
                             "results.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        except Exception as e:
            st.error(f"❌ خطأ: {e}")
