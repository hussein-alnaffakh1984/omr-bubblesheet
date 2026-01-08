import io
import re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw, ImageFont

# Optional but recommended
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_COORDS = True
except ImportError:
    HAS_COORDS = False
    st.warning("⚠️ لتجربة أفضل، ثبّت: pip install streamlit-image-coordinates")


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
    """تحميل صفحات من PDF أو صورة"""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        try:
            pages = convert_from_bytes(file_bytes, dpi=300, fmt="png")
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
    """تحويل PIL إلى BGR لـ OpenCV"""
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """تحويل BGR إلى PIL"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_to(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    """تغيير حجم الصورة"""
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)


# =========================
# Alignment (تصحيح الانحراف - مُحسّن)
# =========================
def order_points(pts: np.ndarray) -> np.ndarray:
    """ترتيب 4 نقاط لتكون: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left (أصغر مجموع)
    rect[2] = pts[np.argmax(s)]  # bottom-right (أكبر مجموع)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def find_page_quad(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    إيجاد الرباعي (quadrilateral) الذي يمثل حدود الورقة
    محسّن للكشف الأفضل
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # كشف الحواف
    edges = cv2.Canny(gray, 50, 150)

    # إغلاق الفجوات
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # إيجاد الcontours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    
    # ترتيب حسب المساحة
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    h, w = bgr.shape[:2]
    min_area = 0.15 * (h * w)  # على الأقل 15% من مساحة الصورة

    for c in cnts[:10]:  # نفحص أول 10 contours
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # نبحث عن رباعي
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > min_area:
                pts = approx.reshape(4, 2).astype(np.float32)
                return order_points(pts)

    return None


def warp_to_template(bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    """
    محاذاة الصورة للنموذج (تصحيح المنظور)
    محسّن مع fallback أفضل
    """
    quad = find_page_quad(bgr)
    
    if quad is None:
        # Fallback: تصحيح انحراف بسيط ثم resize
        h, w = bgr.shape[:2]
        
        # محاولة تصحيح انحراف بسيط
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=w//3, maxLineGap=10)
        
        if lines is not None and len(lines) > 5:
            angles = []
            for line in lines[:20]:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            median_angle = np.median(angles)
            
            # إذا كان الانحراف أكثر من 0.5 درجة
            if abs(median_angle) > 0.5:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                bgr = cv2.warpAffine(bgr, M, (w, h), 
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
        
        return resize_to(bgr, tw, th)

    # تطبيق perspective transform
    dst = np.array([
        [0, 0],
        [tw - 1, 0],
        [tw - 1, th - 1],
        [0, th - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(bgr, M, (tw, th), 
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped


# =========================
# Preprocess & Bubble Scoring (محسّن)
# =========================
def preprocess_for_bubbles(bgr: np.ndarray) -> np.ndarray:
    """
    معالجة مسبقة للصورة لكشف الفقاعات
    محسّن للدقة الأفضل
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold (معكوس - المُظلَّل يصبح أبيض)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 8  # معايير محسّنة
    )
    
    # إزالة الضوضاء الصغيرة
    thr = cv2.medianBlur(thr, 3)
    
    # Morphological operations لتحسين الكشف
    kernel = np.ones((2, 2), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    
    return thr


def inner_crop(cell: np.ndarray, margin_ratio: float = 0.25) -> np.ndarray:
    """
    اقتصاص الحواف من الخلية للتركيز على المركز
    زيادة margin_ratio لتجنب حواف الدائرة
    """
    h, w = cell.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    
    # تأكد من عدم الاقتصاص الزائد
    if h - 2*my <= 0 or w - 2*mx <= 0:
        return cell
    
    return cell[my:h - my, mx:w - mx]


def score_cell(bin_cell: np.ndarray) -> float:
    """
    حساب نسبة التظليل في الخلية
    محسّن للدقة
    """
    if bin_cell.size == 0:
        return 0.0
    
    # اقتصاص الحواف
    c = inner_crop(bin_cell, 0.28)  # 28% من كل جانب
    
    if c.size == 0:
        return 0.0
    
    # حساب البيكسلات البيضاء (المُظللة)
    white_pixels = np.sum(c > 0)
    total_pixels = c.shape[0] * c.shape[1]
    
    return float(white_pixels) / float(total_pixels + 1e-9)


def pick_one(scores: List[Tuple[str, float]], min_fill=0.22, min_ratio=1.4):
    """
    اختيار إجابة واحدة من قائمة الخيارات
    محسّن للتعامل مع الحالات المعقدة
    """
    if not scores:
        return "?", "ERROR", 0.0, 0.0
    
    # ترتيب حسب نسبة التظليل (الأكبر أولاً)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0.0

    # فارغ
    if top_s < min_fill:
        return "?", "BLANK", top_s, second_s
    
    # تظليل مزدوج (إذا كان الفرق صغيراً)
    if second_s > min_fill and (top_s / (second_s + 1e-9)) < min_ratio:
        return "!", "DOUBLE", top_s, second_s
    
    # إجابة واضحة
    return top_c, "OK", top_s, second_s


# =========================
# Read Student Code (محسّن)
# =========================
def read_student_code(thr: np.ndarray, cfg: TemplateConfig) -> Tuple[str, Dict]:
    """
    قراءة كود الطالب من منطقة ID
    محسّن مع معالجة أفضل للأخطاء
    """
    x, y, w, h = cfg.id_roi
    if w <= 0 or h <= 0:
        return "", {"error": "ID ROI not configured"}

    # التأكد من أن ROI ضمن حدود الصورة
    img_h, img_w = thr.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return "", {"error": "ID ROI out of bounds"}

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
            y_start = r * ch
            y_end = (r + 1) * ch
            x_start = c * cw
            x_end = (c + 1) * cw
            
            cell = roi[y_start:y_end, x_start:x_end]
            fill = score_cell(cell)
            col_scores.append((str(r), fill))
        
        # معايير أكثر تساهلاً لكود الطالب
        d, status, top, second = pick_one(col_scores, min_fill=0.20, min_ratio=1.3)
        
        # إذا كان فارغاً أو مزدوجاً، نضع X
        if d in ("?", "!"):
            digits.append("X")
        else:
            digits.append(d)
        
        debug_cols.append({
            "col": c, 
            "status": status, 
            "top": f"{top:.3f}", 
            "second": f"{second:.3f}",
            "digit": d
        })

    code = "".join(digits)
    
    # إذا كان الكود يحتوي على X، نحاول استبداله بأقرب رقم
    # أو نتركه كما هو للمراجعة اليدوية
    
    return code, {"cols": debug_cols, "raw": digits}


# =========================
# Read Answers (محسّن)
# =========================
def read_answers(thr: np.ndarray, block: QBlock, choices: int) -> Dict[int, Tuple[str, str, float, float]]:
    """
    قراءة الإجابات من بلوك أسئلة
    محسّن مع معلومات إضافية للتشخيص
    """
    letters = "ABCDEFGH"[:choices]
    out = {}

    x, y, w, h = block.x, block.y, block.w, block.h
    
    # التأكد من أن البلوك ضمن حدود الصورة
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
            y_start = r * rh
            y_end = (r + 1) * rh
            x_start = c * cw
            x_end = (c + 1) * cw
            
            cell = roi[y_start:y_end, x_start:x_end]
            fill_score = score_cell(cell)
            scores.append((letters[c], fill_score))
        
        a, status, top, second = pick_one(scores, min_fill=0.22, min_ratio=1.4)
        out[q] = (a, status, top, second)
        q += 1
    
    return out


# =========================
# Ranges
# =========================
def parse_ranges(txt: str) -> List[Tuple[int, int]]:
    """تحليل نصوص النطاقات مثل: 1-40, 50-60"""
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
    """التحقق من وجود السؤال في النطاقات"""
    if not ranges:
        return False
    return any(a <= q <= b for a, b in ranges)


# =========================
# Draw Preview (محسّن)
# =========================
def draw_cfg_preview(img: Image.Image, cfg: TemplateConfig, show_grid: bool = False) -> Image.Image:
    """
    رسم preview للإعدادات على الصورة
    محسّن مع ألوان أوضح وخطوط أعرض
    """
    im = img.copy().convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")

    # ID ROI باللون الأحمر
    x, y, w, h = cfg.id_roi
    if w > 0 and h > 0:
        # خلفية شفافة
        dr.rectangle([x, y, x + w, y + h], 
                    fill=(255, 0, 0, 40), 
                    outline=(255, 0, 0), 
                    width=5)
        
        # نص
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        dr.text((x + 10, y + 10), "ID CODE", fill=(255, 255, 255), font=font)
        
        # رسم الشبكة إذا طُلب
        if show_grid and cfg.id_rows > 0 and cfg.id_digits > 0:
            ch = h // cfg.id_rows
            cw = w // cfg.id_digits
            for r in range(1, cfg.id_rows):
                dr.line([x, y + r*ch, x + w, y + r*ch], fill=(255, 100, 100, 128), width=1)
            for c in range(1, cfg.id_digits):
                dr.line([x + c*cw, y, x + c*cw, y + h], fill=(255, 100, 100, 128), width=1)

    # Q blocks باللون الأخضر
    for i, b in enumerate(cfg.q_blocks or [], 1):
        dr.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], 
                    fill=(0, 200, 0, 40), 
                    outline=(0, 200, 0), 
                    width=5)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        dr.text((b.x + 10, b.y + 10), 
               f"Q{i}: {b.start_q}-{b.end_q} ({b.rows}r)", 
               fill=(255, 255, 255), 
               font=font)
        
        # رسم الشبكة
        if show_grid and b.rows > 0:
            rh = b.h // b.rows
            cw = b.w // cfg.choices
            for r in range(1, b.rows):
                dr.line([b.x, b.y + r*rh, b.x + b.w, b.y + r*rh], 
                       fill=(100, 255, 100, 128), width=1)
            for c in range(1, cfg.choices):
                dr.line([b.x + c*cw, b.y, b.x + c*cw, b.y + b.h], 
                       fill=(100, 255, 100, 128), width=1)
    
    return im


# =========================
# UI
# =========================
st.set_page_config(page_title="OMR Bubble Sheet - Remark Style", layout="wide")

st.markdown(
    """
    <style>
      .small-note {opacity:0.8; font-size: 0.9rem; color: #666;}
      .block-title {font-weight:800; font-size:1.25rem; color: #1f77b4; margin-top: 20px; margin-bottom: 10px;}
      .stButton>button {border-radius: 8px; font-weight: 600;}
      .success-box {background: #d4edda; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745; margin: 10px 0;}
      .error-box {background: #f8d7da; padding: 15px; border-radius: 8px; border-left: 5px solid #dc3545; margin: 10px 0;}
      .info-box {background: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 5px solid #17a2b8; margin: 10px 0;}
      .warning-box {background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 5px solid #ffc107; margin: 10px 0;}
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("✅ OMR Bubble Sheet Scanner — Remark-Style System")
st.markdown("---")

# Session state initialization
if "cfg" not in st.session_state:
    st.session_state.cfg = TemplateConfig(q_blocks=[])

if "clicks" not in st.session_state:
    st.session_state.clicks = []

if "template_img" not in st.session_state:
    st.session_state.template_img = None

if "template_bytes" not in st.session_state:
    st.session_state.template_bytes = None

if "template_name" not in st.session_state:
    st.session_state.template_name = ""

if "show_grid" not in st.session_state:
    st.session_state.show_grid = False

if "results_df" not in st.session_state:
    st.session_state.results_df = None


# =========================
# Layout: Left = Canvas, Right = Controls
# =========================
left, right = st.columns([1.6, 1], gap="large")

# =========================
# RIGHT PANEL: Controls
# =========================
with right:
    st.markdown('<div class="block-title">📄 1) رفع نموذج الورقة (Template)</div>', unsafe_allow_html=True)
    tpl = st.file_uploader(
        "PDF/PNG/JPG (الصفحة الأولى ستستخدم كنموذج)", 
        type=["pdf", "png", "jpg", "jpeg"], 
        key="tpl_upl",
        help="ارفع نموذج البابل شيت الفارغ"
    )

    if tpl is not None:
        st.session_state.template_bytes = tpl.getvalue()
        st.session_state.template_name = tpl.name
        
        with st.spinner("⏳ جاري تحميل النموذج..."):
            pages = load_pages(st.session_state.template_bytes, st.session_state.template_name)
            
            if pages:
                st.session_state.template_img = pages[0].convert("RGB")
                tw, th = st.session_state.template_img.size
                st.session_state.cfg.template_w = tw
                st.session_state.cfg.template_h = th
                st.success(f"✅ تم تحميل النموذج ({tw}x{th})")
            else:
                st.error("❌ فشل تحميل النموذج")
                st.stop()

    st.markdown('<div class="block-title">⚙️ 2) إعدادات عامة</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        canvas_w = st.slider("عرض المعاينة", 500, 1400, 800, 50, help="عرض الصورة في Canvas")
    with col2:
        st.session_state.show_grid = st.checkbox("إظهار الشبكة", value=st.session_state.show_grid, help="عرض خطوط الشبكة على الصورة")
    
    col3, col4 = st.columns(2)
    with col3:
        choices = st.selectbox("عدد الخيارات", [4, 5, 6], index=0, help="عدد خيارات كل سؤال (A,B,C,D...)")
    with col4:
        id_digits = st.number_input("خانات كود الطالب", 1, 12, int(st.session_state.cfg.id_digits), 1, help="عدد أرقام كود الطالب")
    
    id_rows = st.number_input("صفوف أرقام الكود", 5, 15, int(st.session_state.cfg.id_rows), 1, help="عادةً 10 (من 0-9)")

    st.session_state.cfg.choices = int(choices)
    st.session_state.cfg.id_digits = int(id_digits)
    st.session_state.cfg.id_rows = int(id_rows)

    st.markdown('<div class="block-title">🎯 3) إعداد البلوكات</div>', unsafe_allow_html=True)
    
    mode = st.radio(
        "ماذا نحدد الآن؟",
        ["🆔 ID ROI (منطقة كود الطالب)", "📝 Q Block (بلوك الأسئلة)"],
        index=0,
        help="اختر نوع المنطقة التي تريد تحديدها"
    )

    if mode.startswith("📝"):
        col5, col6, col7 = st.columns(3)
        with col5:
            b_start = st.number_input("Start Q", 1, 500, 1, 1)
        with col6:
            b_end = st.number_input("End Q", 1, 500, 20, 1)
        with col7:
            b_rows = st.number_input("Rows", 1, 200, 20, 1)
    else:
        b_start = b_end = b_rows = 0

    st.markdown('<div class="info-box">💡 اضغط نقطتين على الصورة (يسار): الزاوية الأولى ثم الزاوية الثانية</div>', unsafe_allow_html=True)

    col8, col9, col10 = st.columns(3)
    with col8:
        if st.button("↶ مسح آخر نقطة", use_container_width=True):
            if st.session_state.clicks:
                st.session_state.clicks.pop()
                st.success("✅ تم المسح")
    
    with col9:
        if st.button("🔄 Reset الكل", use_container_width=True):
            st.session_state.clicks = []
            st.session_state.cfg.id_roi = (0, 0, 0, 0)
            st.session_state.cfg.q_blocks = []
            st.success("✅ تم Reset")
    
    with col10:
        st.write("")  # spacer

    if st.button("💾 حفظ المستطيل الحالي", use_container_width=True, type="primary"):
        if len(st.session_state.clicks) < 2:
            st.error("❌ يجب اختيار نقطتين أولاً")
        else:
            (x1, y1), (x2, y2) = st.session_state.clicks[-2], st.session_state.clicks[-1]
            x = int(min(x1, x2))
            y = int(min(y1, y2))
            w = int(abs(x2 - x1))
            h = int(abs(y2 - y1))

            if w < 10 or h < 10:
                st.error("❌ المستطيل صغير جداً (على الأقل 10x10 بيكسل)")
            else:
                if mode.startswith("🆔"):
                    st.session_state.cfg.id_roi = (x, y, w, h)
                    st.markdown('<div class="success-box">✅ تم حفظ ID ROI بنجاح</div>', unsafe_allow_html=True)
                else:
                    qb = QBlock(
                        x=x, y=y, w=w, h=h,
                        start_q=int(min(b_start, b_end)),
                        end_q=int(max(b_start, b_end)),
                        rows=int(b_rows)
                    )
                    st.session_state.cfg.q_blocks.append(qb)
                    st.markdown(f'<div class="success-box">✅ تم إضافة Q Block: أسئلة {qb.start_q}-{qb.end_q}</div>', unsafe_allow_html=True)
                
                # مسح آخر نقطتين
                if len(st.session_state.clicks) >= 2:
                    st.session_state.clicks = st.session_state.clicks[:-2]

    if st.button("🗑️ حذف آخر Q Block", use_container_width=True):
        if st.session_state.cfg.q_blocks:
            removed = st.session_state.cfg.q_blocks.pop()
            st.success(f"✅ تم حذف البلوك: {removed.start_q}-{removed.end_q}")
        else:
            st.info("ℹ️ لا يوجد بلوكات للحذف")

    # عرض البلوكات الحالية
    if st.session_state.cfg.q_blocks:
        st.markdown("**البلوكات الحالية:**")
        for i, b in enumerate(st.session_state.cfg.q_blocks, 1):
            st.text(f"{i}. Q{b.start_q}-{b.end_q} ({b.rows} صفوف)")

    st.markdown('<div class="block-title">📂 4) ملفات التصحيح</div>', unsafe_allow_html=True)
    
    roster_file = st.file_uploader(
        "📋 Roster (Excel/CSV)",
        type=["xlsx", "xls", "csv"],
        key="roster_upl",
        help="ملف يحتوي على: student_code و student_name"
    )
    
    key_file = st.file_uploader(
        "🔑 Answer Key (نموذج الإجابات)",
        type=["pdf", "png", "jpg", "jpeg"],
        key="key_upl",
        help="نفس النموذج مع الإجابات الصحيحة مُعلمة"
    )
    
    sheets_file = st.file_uploader(
        "📚 أوراق الطلاب (PDF/صور)",
        type=["pdf", "png", "jpg", "jpeg"],
        key="sheets_upl",
        help="ملف PDF يحتوي على جميع أوراق الطلاب"
    )

    st.markdown('<div class="block-title">📊 5) نطاقات الدرجات</div>', unsafe_allow_html=True)
    
    theory_txt = st.text_input(
        "النطاق النظري",
        "",
        placeholder="مثال: 1-40 أو 1-20,25-40",
        help="حدد نطاق الأسئلة النظرية"
    )
    
    practical_txt = st.text_input(
        "النطاق العملي (اختياري)",
        "",
        placeholder="مثال: 41-60",
        help="حدد نطاق الأسئلة العملية إن وجد"
    )
    
    col11, col12 = st.columns(2)
    with col11:
        strict = st.checkbox("وضع صارم", True, help="BLANK/DOUBLE = خطأ")
    with col12:
        min_fill = st.slider("حد التظليل", 0.15, 0.35, 0.22, 0.01, help="الحد الأدنى للتظليل")


# =========================
# LEFT PANEL: Canvas
# =========================
with left:
    if st.session_state.template_img is None:
        st.info("📄 ارفع Template من اليمين للبدء")
        st.stop()

    # رسم preview مع الإعدادات
    preview = draw_cfg_preview(
        st.session_state.template_img, 
        st.session_state.cfg,
        show_grid=st.session_state.show_grid
    )

    st.markdown('<div class="block-title">🖱️ واجهة التحديد (اضغط نقطتين)</div>', unsafe_allow_html=True)

    if HAS_COORDS:
        # استخدام streamlit_image_coordinates
        coords = streamlit_image_coordinates(preview, width=canvas_w, key="img_coords")

        if coords is not None and "x" in coords and "y" in coords:
            # تحويل من حجم العرض للحجم الأصلي
            orig_w, orig_h = st.session_state.template_img.size
            scale = orig_w / float(canvas_w)
            x = int(coords["x"] * scale)
            y = int(coords["y"] * scale)
            
            # إضافة النقطة
            st.session_state.clicks.append((x, y))
            
            # رسم النقطة على الصورة
            draw = ImageDraw.Draw(preview)
            # حساب موقع النقطة على الصورة المعروضة
            display_x = coords["x"]
            display_y = coords["y"]
            r = 8
            draw.ellipse([display_x-r, display_y-r, display_x+r, display_y+r], 
                        fill=(255, 0, 0), outline=(255, 255, 255), width=2)
            
            st.success(f"📍 نقطة جديدة: ({x}, {y})")
    else:
        # Fallback: عرض الصورة فقط
        st.image(preview, width=canvas_w, use_column_width=False)
        st.warning("⚠️ لتحديد النقاط تلقائياً، ثبّت: pip install streamlit-image-coordinates")
        
        # إدخال يدوي
        col_x, col_y = st.columns(2)
        with col_x:
            manual_x = st.number_input("X", 0, st.session_state.cfg.template_w, 0)
        with col_y:
            manual_y = st.number_input("Y", 0, st.session_state.cfg.template_h, 0)
        
        if st.button("➕ إضافة نقطة يدوياً"):
            st.session_state.clicks.append((int(manual_x), int(manual_y)))
            st.success(f"✅ تمت الإضافة: ({manual_x}, {manual_y})")

    # عرض المستطيل الحالي
    if len(st.session_state.clicks) >= 2:
        (x1, y1), (x2, y2) = st.session_state.clicks[-2], st.session_state.clicks[-1]
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        
        st.markdown(f'<div class="info-box">🎯 المستطيل الحالي: x={x}, y={y}, w={w}, h={h}</div>', unsafe_allow_html=True)

    # عرض جميع النقاط
    if st.session_state.clicks:
        with st.expander(f"📍 النقاط المحددة ({len(st.session_state.clicks)})"):
            for i, (x, y) in enumerate(st.session_state.clicks, 1):
                st.text(f"{i}. ({x}, {y})")

    st.markdown("---")

    # =========================
    # GRADING SECTION
    # =========================
    st.markdown('<div class="block-title">🚀 التصحيح</div>', unsafe_allow_html=True)
    
    if st.button("🎯 ابدأ التصحيح الآن", use_container_width=True, type="primary"):
        cfg = st.session_state.cfg

        # Validation
        errors = []
        
        if cfg.template_w <= 0 or cfg.template_h <= 0:
            errors.append("❌ النموذج غير صالح")
        
        if cfg.id_roi[2] <= 0 or cfg.id_roi[3] <= 0:
            errors.append("❌ يجب تحديد ID ROI")
        
        if not cfg.q_blocks:
            errors.append("❌ يجب إضافة بلوك أسئلة واحد على الأقل")
        
        if roster_file is None:
            errors.append("❌ يجب رفع ملف Roster")
        
        if key_file is None:
            errors.append("❌ يجب رفع Answer Key")
        
        if sheets_file is None:
            errors.append("❌ يجب رفع أوراق الطلاب")
        
        if errors:
            for err in errors:
                st.error(err)
            st.stop()

        # Start grading
        st.markdown('<div class="success-box">✅ بدء عملية التصحيح...</div>', unsafe_allow_html=True)

        try:
            # Load roster
            with st.spinner("📋 قراءة Roster..."):
                if roster_file.name.lower().endswith(("xlsx", "xls")):
                    df_roster = pd.read_excel(roster_file)
                else:
                    df_roster = pd.read_csv(roster_file)

                df_roster.columns = [c.strip().lower().replace(" ", "_") for c in df_roster.columns]
                
                if "student_code" not in df_roster.columns or "student_name" not in df_roster.columns:
                    st.error("❌ Roster يجب أن يحتوي على: student_code و student_name")
                    st.stop()

                roster = dict(
                    zip(
                        df_roster["student_code"].astype(str).str.strip(),
                        df_roster["student_name"].astype(str).str.strip()
                    )
                )
                
                st.success(f"✅ تم تحميل {len(roster)} طالب من Roster")

            # Load and process answer key
            with st.spinner("🔑 معالجة Answer Key..."):
                key_pages = load_pages(key_file.getvalue(), key_file.name)
                if not key_pages:
                    st.error("❌ فشل تحميل Answer Key")
                    st.stop()
                
                key_bgr = pil_to_bgr(key_pages[0])
                key_bgr = warp_to_template(key_bgr, cfg.template_w, cfg.template_h)
                key_thr = preprocess_for_bubbles(key_bgr)

                # Read key answers
                key_ans = {}
                for b in cfg.q_blocks:
                    block_ans = read_answers(key_thr, b, cfg.choices)
                    for q, (ans, status, top, second) in block_ans.items():
                        key_ans[q] = ans
                
                st.success(f"✅ تم قراءة {len(key_ans)} سؤال من Answer Key")
                
                # عرض عينة من الإجابات
                sample = list(key_ans.items())[:10]
                st.text("عينة من الإجابات: " + ", ".join([f"Q{q}:{a}" for q, a in sample]))

            # Parse ranges
            theory_ranges = parse_ranges(theory_txt)
            practical_ranges = parse_ranges(practical_txt)
            
            if theory_ranges:
                st.info(f"📊 النطاق النظري: {theory_ranges}")
            if practical_ranges:
                st.info(f"📊 النطاق العملي: {practical_ranges}")

            # Load student sheets
            with st.spinner("📚 تحميل أوراق الطلاب..."):
                pages = load_pages(sheets_file.getvalue(), sheets_file.name)
                if not pages:
                    st.error("❌ فشل تحميل أوراق الطلاب")
                    st.stop()
                
                st.success(f"✅ تم تحميل {len(pages)} ورقة")

            # Process sheets
            prog_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total_pages = len(pages)
            
            detailed_results = []  # للتفاصيل

            for idx, pg in enumerate(pages, 1):
                status_text.text(f"⏳ معالجة ورقة {idx}/{total_pages}...")
                
                try:
                    # Convert and warp
                    bgr = pil_to_bgr(pg)
                    bgr = warp_to_template(bgr, cfg.template_w, cfg.template_h)
                    thr = preprocess_for_bubbles(bgr)

                    # Read student code
                    code, code_dbg = read_student_code(thr, cfg)
                    code = (code or "").strip().replace("X", "")
                    
                    if code == "":
                        code = f"UNKNOWN_{idx}"

                    name = roster.get(code, "غير موجود في Roster")

                    # Read student answers
                    stu_ans = {}
                    for b in cfg.q_blocks:
                        block_ans = read_answers(thr, b, cfg.choices)
                        stu_ans.update(block_ans)

                    # Calculate scores
                    score_theory = 0
                    total_theory = 0
                    score_practical = 0
                    total_practical = 0
                    score_total = 0
                    total_total = 0
                    
                    details = []

                    for q, key_ans_val in key_ans.items():
                        # Determine if question is in ranges
                        in_theory = theory_ranges and in_ranges(q, theory_ranges)
                        in_practical = practical_ranges and in_ranges(q, practical_ranges)
                        
                        # If no ranges specified, count all
                        if not theory_ranges and not practical_ranges:
                            in_theory = True
                            in_practical = False

                        if not (in_theory or in_practical):
                            continue

                        student_ans, status, top, second = stu_ans.get(q, ("?", "MISSING", 0, 0))

                        # Strict mode handling
                        if strict and status in ("BLANK", "DOUBLE"):
                            is_correct = False
                        else:
                            is_correct = (student_ans == key_ans_val)

                        # Update scores
                        if in_theory:
                            total_theory += 1
                            if is_correct:
                                score_theory += 1
                        
                        if in_practical:
                            total_practical += 1
                            if is_correct:
                                score_practical += 1
                        
                        total_total += 1
                        if is_correct:
                            score_total += 1
                        
                        details.append({
                            "question": q,
                            "key": key_ans_val,
                            "student": student_ans,
                            "status": status,
                            "correct": is_correct
                        })

                    # Calculate percentages
                    pct_theory = (score_theory / total_theory * 100) if total_theory > 0 else 0
                    pct_practical = (score_practical / total_practical * 100) if total_practical > 0 else 0
                    pct_total = (score_total / total_total * 100) if total_total > 0 else 0

                    results.append({
                        "sheet_index": idx,
                        "student_code": code,
                        "student_name": name,
                        "theory_score": score_theory,
                        "theory_total": total_theory,
                        "theory_pct": f"{pct_theory:.2f}%",
                        "practical_score": score_practical,
                        "practical_total": total_practical,
                        "practical_pct": f"{pct_practical:.2f}%",
                        "total_score": score_total,
                        "total_questions": total_total,
                        "total_pct": f"{pct_total:.2f}%"
                    })
                    
                    detailed_results.append({
                        "code": code,
                        "name": name,
                        "details": details
                    })

                except Exception as e:
                    st.warning(f"⚠️ خطأ في ورقة {idx}: {str(e)}")
                    results.append({
                        "sheet_index": idx,
                        "student_code": f"ERROR_{idx}",
                        "student_name": "خطأ في المعالجة",
                        "theory_score": 0,
                        "theory_total": 0,
                        "theory_pct": "0%",
                        "practical_score": 0,
                        "practical_total": 0,
                        "practical_pct": "0%",
                        "total_score": 0,
                        "total_questions": 0,
                        "total_pct": "0%"
                    })

                prog_bar.progress(int(idx / total_pages * 100))

            status_text.empty()
            prog_bar.empty()

            # Create DataFrame
            df_results = pd.DataFrame(results)
            st.session_state.results_df = df_results

            # Display results
            st.markdown('<div class="success-box">✅ اكتمل التصحيح بنجاح!</div>', unsafe_allow_html=True)
            
            st.markdown("### 📊 النتائج")
            st.dataframe(df_results, use_container_width=True, height=400)

            # Statistics
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("إجمالي الأوراق", len(df_results))
            with col_s2:
                avg_score = df_results['total_pct'].str.rstrip('%').astype(float).mean()
                st.metric("متوسط الدرجات", f"{avg_score:.2f}%")
            with col_s3:
                passed = (df_results['total_pct'].str.rstrip('%').astype(float) >= 50).sum()
                st.metric("الناجحين (≥50%)", passed)

            # Export Excel
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df_results.to_excel(writer, index=False, sheet_name="Results")
                
                # Add detailed sheet
                if detailed_results:
                    detailed_rows = []
                    for dr in detailed_results:
                        for det in dr["details"]:
                            detailed_rows.append({
                                "student_code": dr["code"],
                                "student_name": dr["name"],
                                "question": det["question"],
                                "key_answer": det["key"],
                                "student_answer": det["student"],
                                "status": det["status"],
                                "correct": "✓" if det["correct"] else "✗"
                            })
                    
                    df_detailed = pd.DataFrame(detailed_rows)
                    df_detailed.to_excel(writer, index=False, sheet_name="Detailed")

            st.download_button(
                "⬇️ تحميل النتائج (Excel)",
                buf.getvalue(),
                "bubble_sheet_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ خطأ في التصحيح: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # Show previous results if available
    if st.session_state.results_df is not None:
        st.markdown("---")
        st.markdown("### 📋 آخر نتائج")
        st.dataframe(st.session_state.results_df, use_container_width=True, height=300)
