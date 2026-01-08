"""
======================================================================================
                    OMR BUBBLE SHEET SCANNER - TRUE REMARK STYLE
                         نظام تصحيح البابل شيت - نسخة Remark الحقيقية
======================================================================================
✅ Drag & Drop مباشر مثل Remark تماماً
"""

import io
from dataclasses import dataclass
from typing import List, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw


# ======================================================================================
#                                   DATA MODELS
# ======================================================================================

@dataclass
class Rectangle:
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass
class QuestionBlock:
    rect: Rectangle
    start_q: int
    end_q: int
    num_rows: int


@dataclass
class Template:
    width: int
    height: int
    id_block: Optional[Rectangle] = None
    q_blocks: List[QuestionBlock] = None
    num_choices: int = 4
    id_digits: int = 4
    id_rows: int = 10
    
    def __post_init__(self):
        if self.q_blocks is None:
            self.q_blocks = []


# ======================================================================================
#                              IMAGE PROCESSING
# ======================================================================================

class ImageProcessor:
    
    @staticmethod
    def load_image(file_bytes: bytes, filename: str) -> Optional[Image.Image]:
        try:
            if filename.lower().endswith('.pdf'):
                pages = convert_from_bytes(file_bytes, dpi=200)
                return pages[0].convert('RGB') if pages else None
            return Image.open(io.BytesIO(file_bytes)).convert('RGB')
        except Exception as e:
            st.error(f"خطأ: {e}")
            return None
    
    @staticmethod
    def align_and_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def preprocess(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 6)
        return binary


# ======================================================================================
#                              BUBBLE DETECTION
# ======================================================================================

class BubbleDetector:
    
    def __init__(self, min_fill: float = 0.20):
        self.min_fill = min_fill
    
    def calculate_fill(self, cell: np.ndarray) -> float:
        if cell.size == 0:
            return 0.0
        
        h, w = cell.shape[:2]
        margin_h = int(h * 0.25)
        margin_w = int(w * 0.25)
        
        if h - 2*margin_h <= 0 or w - 2*margin_w <= 0:
            return 0.0
        
        inner = cell[margin_h:h-margin_h, margin_w:w-margin_w]
        return np.sum(inner > 0) / inner.size if inner.size > 0 else 0.0
    
    def detect_answer(self, cells: List[np.ndarray], choices: List[str]) -> Dict:
        fills = [self.calculate_fill(c) for c in cells]
        sorted_idx = sorted(range(len(fills)), key=lambda i: fills[i], reverse=True)
        
        top_idx = sorted_idx[0]
        top_fill = fills[top_idx]
        second_fill = fills[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0
        
        if top_fill < self.min_fill:
            return {"answer": "?", "status": "BLANK"}
        
        if second_fill > self.min_fill and (top_fill / (second_fill + 1e-9)) < 1.4:
            return {"answer": "!", "status": "DOUBLE"}
        
        return {"answer": choices[top_idx], "status": "OK"}


# ======================================================================================
#                                GRADING ENGINE
# ======================================================================================

class GradingEngine:
    
    def __init__(self, template: Template):
        self.template = template
        self.detector = BubbleDetector()
    
    def extract_id(self, binary: np.ndarray) -> str:
        if not self.template.id_block:
            return ""
        
        rect = self.template.id_block
        h, w = binary.shape[:2]
        
        if rect.x < 0 or rect.y < 0 or rect.x2 > w or rect.y2 > h:
            return "OUT_OF_BOUNDS"
        
        roi = binary[rect.y:rect.y2, rect.x:rect.x2]
        
        rows = self.template.id_rows
        cols = self.template.id_digits
        cell_h = rect.height // rows
        cell_w = rect.width // cols
        
        digits = []
        for col in range(cols):
            col_cells = []
            for row in range(rows):
                cell = roi[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w]
                col_cells.append(cell)
            
            result = self.detector.detect_answer(col_cells, [str(i) for i in range(10)])
            digits.append(result["answer"] if result["status"] == "OK" else "X")
        
        return "".join(digits)
    
    def extract_answers(self, binary: np.ndarray, block: QuestionBlock) -> Dict:
        rect = block.rect
        h, w = binary.shape[:2]
        
        if rect.x < 0 or rect.y < 0 or rect.x2 > w or rect.y2 > h:
            return {}
        
        roi = binary[rect.y:rect.y2, rect.x:rect.x2]
        
        rows = block.num_rows
        cols = self.template.num_choices
        cell_h = rect.height // rows
        cell_w = rect.width // cols
        
        choices = "ABCDEFGH"[:self.template.num_choices]
        answers = {}
        
        q_num = block.start_q
        for row in range(rows):
            if q_num > block.end_q:
                break
            
            row_cells = []
            for col in range(cols):
                cell = roi[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w]
                row_cells.append(cell)
            
            result = self.detector.detect_answer(row_cells, list(choices))
            answers[q_num] = result
            q_num += 1
        
        return answers
    
    def grade_sheet(self, img: np.ndarray, answer_key: Dict, roster: Dict, 
                   strict: bool = True) -> Dict:
        aligned = ImageProcessor.align_and_resize(img, self.template.width, 
                                                 self.template.height)
        binary = ImageProcessor.preprocess(aligned)
        
        student_id = self.extract_id(binary)
        student_name = roster.get(student_id, "غير موجود")
        
        all_answers = {}
        for block in self.template.q_blocks:
            all_answers.update(self.extract_answers(binary, block))
        
        correct = 0
        total = len(answer_key)
        
        for q, correct_ans in answer_key.items():
            if q not in all_answers:
                continue
            
            student_result = all_answers[q]
            if strict and student_result["status"] != "OK":
                continue
            
            if student_result["answer"] == correct_ans:
                correct += 1
        
        percentage = (correct / total * 100) if total > 0 else 0
        
        return {
            "id": student_id,
            "name": student_name,
            "score": correct,
            "total": total,
            "percentage": percentage,
            "passed": percentage >= 50
        }


# ======================================================================================
#                                    UI - REMARK STYLE
# ======================================================================================

def draw_preview(img: Image.Image, template: Template) -> Image.Image:
    preview = img.copy()
    draw = ImageDraw.Draw(preview)
    
    if template.id_block:
        r = template.id_block
        draw.rectangle([r.x, r.y, r.x2, r.y2], outline="red", width=4)
        draw.text((r.x+10, r.y+10), "ID CODE", fill="red")
    
    for i, block in enumerate(template.q_blocks, 1):
        r = block.rect
        draw.rectangle([r.x, r.y, r.x2, r.y2], outline="green", width=4)
        draw.text((r.x+10, r.y+10), f"Q{block.start_q}-{block.end_q}", fill="green")
    
    return preview


def main():
    st.set_page_config(page_title="OMR Remark Style", layout="wide", initial_sidebar_state="collapsed")
    
    # Custom CSS - Remark style
    st.markdown("""
    <style>
        .block-container {padding: 1rem 2rem;}
        .stApp {background: #f5f5f5;}
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
        }
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .step-number {
            background: #667eea;
            color: white;
            border-radius: 50%;
            width: 35px;
            height: 35px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✅ نظام تصحيح البابل شيت - Remark Style</h1>
        <p>تحديد سهل وسريع بالطريقة المرئية</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Session State
    if "template" not in st.session_state:
        st.session_state.template = None
    if "template_img" not in st.session_state:
        st.session_state.template_img = None
    if "current_region" not in st.session_state:
        st.session_state.current_region = None
    
    # ==========================================
    # STEP 1: Upload Template
    # ==========================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">1</span>**رفع نموذج البابل شيت**', unsafe_allow_html=True)
    
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        template_file = st.file_uploader(
            "اختر ملف النموذج",
            type=["pdf", "png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
    
    with col_info:
        if template_file:
            img = ImageProcessor.load_image(template_file.getvalue(), template_file.name)
            if img:
                st.session_state.template_img = img
                w, h = img.size
                
                if st.session_state.template is None:
                    st.session_state.template = Template(w, h)
                else:
                    st.session_state.template.width = w
                    st.session_state.template.height = h
                
                st.success(f"✅ تم التحميل\n{w} × {h} بكسل")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not st.session_state.template_img:
        st.info("👆 ابدأ برفع نموذج البابل شيت")
        st.stop()
    
    # ==========================================
    # STEP 2: Settings
    # ==========================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">2</span>**الإعدادات الأساسية**', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        choices = st.selectbox("عدد الخيارات", [4, 5, 6], 0)
        st.session_state.template.num_choices = choices
    
    with col2:
        id_digits = st.number_input("خانات الكود", 1, 12, 4, 1)
        st.session_state.template.id_digits = id_digits
    
    with col3:
        id_rows = st.number_input("صفوف الكود", 5, 15, 10, 1)
        st.session_state.template.id_rows = id_rows
    
    with col4:
        image_scale = st.slider("حجم العرض", 50, 150, 100, 10)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ==========================================
    # STEP 3: Define Regions - REMARK WAY!
    # ==========================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">3</span>**تحديد المناطق - طريقة Remark**', unsafe_allow_html=True)
    
    col_mode, col_params = st.columns([1, 2])
    
    with col_mode:
        region_type = st.radio(
            "اختر نوع المنطقة:",
            ["🆔 منطقة كود الطالب", "📝 بلوك الأسئلة"],
            label_visibility="collapsed"
        )
    
    with col_params:
        if region_type == "📝 بلوك الأسئلة":
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                start_q = st.number_input("من سؤال", 1, 500, 1, key="start")
            with col_b:
                end_q = st.number_input("إلى سؤال", 1, 500, 20, key="end")
            with col_c:
                num_rows = st.number_input("عدد الصفوف", 1, 200, 20, key="rows")
        else:
            start_q = end_q = num_rows = 0
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ==========================================
    # INTERACTIVE IMAGE - REMARK STYLE!
    # ==========================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🖼️ الصورة التفاعلية")
    
    # Draw preview
    preview = draw_preview(st.session_state.template_img, st.session_state.template)
    
    # Calculate display size
    orig_w, orig_h = preview.size
    display_w = int(orig_w * image_scale / 100)
    display_h = int(orig_h * image_scale / 100)
    
    # Show image
    st.image(preview, width=display_w)
    
    st.markdown("---")
    
    # Simple coordinate input - CLEAREST WAY
    st.markdown("### 📐 طريقة Remark البسيطة:")
    st.info("""
    **كيف تحصل على الإحداثيات:**
    
    1️⃣ **افتح الصورة في Paint** (كليك يمين → فتح باستخدام → Paint)
    
    2️⃣ **ضع الماوس على الزاوية العلوية اليسرى** للمنطقة المطلوبة
       → انظر أسفل الشاشة، ستجد: `80px, 200px`
    
    3️⃣ **ضع الماوس على الزاوية السفلية اليمنى**
       → انظر أسفل الشاشة، ستجد: `350px, 450px`
    
    4️⃣ **أدخل هذه الأرقام أدناه** ← تم! 🎉
    """)
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.markdown("**🔵 الزاوية الأولى (أعلى يسار)**")
        x1 = st.number_input("X الأول", 0, orig_w, 0, 10, key="x1_input")
        y1 = st.number_input("Y الأول", 0, orig_h, 0, 10, key="y1_input")
    
    with col_input2:
        st.markdown("**🔵 الزاوية الثانية (أسفل يمين)**")
        x2 = st.number_input("X الثاني", 0, orig_w, 100, 10, key="x2_input")
        y2 = st.number_input("Y الثاني", 0, orig_h, 100, 10, key="y2_input")
    
    # Show calculated rectangle info
    calc_x = min(x1, x2)
    calc_y = min(y1, y2)
    calc_w = abs(x2 - x1)
    calc_h = abs(y2 - y1)
    
    st.info(f"📏 **المستطيل المحسوب:** الموضع ({calc_x}, {calc_y}) | الحجم {calc_w} × {calc_h}")
    
    # Save button - BIG and CLEAR
    if st.button("💾 حفظ المنطقة", type="primary", use_container_width=True):
        if calc_w < 10 or calc_h < 10:
            st.error("❌ المستطيل صغير جداً! يجب أن يكون على الأقل 10×10 بكسل")
        else:
            rect = Rectangle(calc_x, calc_y, calc_w, calc_h)
            
            if region_type == "🆔 منطقة كود الطالب":
                st.session_state.template.id_block = rect
                st.success("✅ تم حفظ منطقة كود الطالب بنجاح!")
            else:
                block = QuestionBlock(rect, start_q, end_q, num_rows)
                st.session_state.template.q_blocks.append(block)
                st.success(f"✅ تم إضافة بلوك الأسئلة ({start_q}-{end_q}) بنجاح!")
            
            st.rerun()
    
    # Show saved regions
    if st.session_state.template.id_block or st.session_state.template.q_blocks:
        st.markdown("---")
        st.markdown("### 📋 المناطق المحفوظة:")
        
        if st.session_state.template.id_block:
            r = st.session_state.template.id_block
            st.success(f"🆔 **منطقة الكود:** ({r.x}, {r.y}) → ({r.x2}, {r.y2})")
        
        for i, block in enumerate(st.session_state.template.q_blocks, 1):
            r = block.rect
            col_block, col_delete = st.columns([4, 1])
            with col_block:
                st.success(f"📝 **بلوك {i}:** أسئلة {block.start_q}-{block.end_q} | ({r.x}, {r.y}) → ({r.x2}, {r.y2})")
            with col_delete:
                if st.button("🗑️ حذف", key=f"delete_{i}"):
                    st.session_state.template.q_blocks.pop(i-1)
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ==========================================
    # STEP 4: Grading Files
    # ==========================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">4</span>**ملفات التصحيح**', unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        roster = st.file_uploader("📋 قائمة الطلاب", type=["xlsx", "csv"])
    
    with col_f2:
        key_file = st.file_uploader("🔑 نموذج الإجابات", type=["pdf", "png", "jpg"])
    
    with col_f3:
        sheets = st.file_uploader("📚 أوراق الطلاب", type=["pdf", "png", "jpg"])
    
    strict = st.checkbox("✓ وضع صارم (BLANK/DOUBLE = خطأ)", True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ==========================================
    # STEP 5: Start Grading
    # ==========================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">5</span>**ابدأ التصحيح**', unsafe_allow_html=True)
    
    if st.button("🚀 ابدأ التصحيح الآن", type="primary", use_container_width=True):
        # Validation
        errors = []
        if not st.session_state.template.id_block:
            errors.append("❌ يجب تحديد منطقة كود الطالب")
        if not st.session_state.template.q_blocks:
            errors.append("❌ يجب إضافة بلوك أسئلة واحد على الأقل")
        if not roster:
            errors.append("❌ يجب رفع قائمة الطلاب")
        if not key_file:
            errors.append("❌ يجب رفع نموذج الإجابات")
        if not sheets:
            errors.append("❌ يجب رفع أوراق الطلاب")
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                with st.spinner("⏳ جاري التصحيح..."):
                    # Load roster
                    if roster.name.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(roster)
                    else:
                        df = pd.read_csv(roster)
                    
                    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                    roster_dict = dict(zip(df["student_code"].astype(str).str.strip(),
                                         df["student_name"].astype(str).str.strip()))
                    
                    st.info(f"📋 تم تحميل {len(roster_dict)} طالب")
                    
                    # Process key
                    key_img = ImageProcessor.load_image(key_file.getvalue(), key_file.name)
                    key_bgr = cv2.cvtColor(np.array(key_img), cv2.COLOR_RGB2BGR)
                    
                    engine = GradingEngine(st.session_state.template)
                    
                    key_aligned = ImageProcessor.align_and_resize(
                        key_bgr,
                        st.session_state.template.width,
                        st.session_state.template.height
                    )
                    key_binary = ImageProcessor.preprocess(key_aligned)
                    
                    answer_key = {}
                    for block in st.session_state.template.q_blocks:
                        answers = engine.extract_answers(key_binary, block)
                        for q, result in answers.items():
                            if result["status"] == "OK":
                                answer_key[q] = result["answer"]
                    
                    st.success(f"✅ تم استخراج {len(answer_key)} إجابة صحيحة")
                    
                    # Grade
                    sheets_img = ImageProcessor.load_image(sheets.getvalue(), sheets.name)
                    sheets_bgr = cv2.cvtColor(np.array(sheets_img), cv2.COLOR_RGB2BGR)
                    
                    result = engine.grade_sheet(sheets_bgr, answer_key, roster_dict, strict)
                    
                    st.success("✅ اكتمل التصحيح بنجاح!")
                    
                    # Display results
                    df_results = pd.DataFrame([{
                        "كود الطالب": result["id"],
                        "اسم الطالب": result["name"],
                        "الإجابات الصحيحة": result["score"],
                        "إجمالي الأسئلة": result["total"],
                        "النسبة المئوية": f"{result['percentage']:.1f}%",
                        "الحالة": "ناجح ✓" if result["passed"] else "راسب ✗"
                    }])
                    
                    st.dataframe(df_results, use_container_width=True, height=150)
                    
                    # Export
                    buffer = io.BytesIO()
                    df_results.to_excel(buffer, index=False, engine='openpyxl')
                    
                    st.download_button(
                        "⬇️ تحميل النتائج (Excel)",
                        buffer.getvalue(),
                        "results.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            except Exception as e:
                st.error(f"❌ حدث خطأ: {e}")
                import traceback
                with st.expander("عرض تفاصيل الخطأ"):
                    st.code(traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
