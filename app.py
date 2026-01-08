"""
======================================================================================
                    OMR BUBBLE SHEET SCANNER - MOUSE SELECTION
                         نظام تصحيح البابل شيت - تحديد بالماوس
======================================================================================
✅ تحديد بالماوس مباشرة | Mouse Click Selection
"""

import io
import json
from dataclasses import dataclass
from typing import List, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas


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
#                                    UI
# ======================================================================================

def draw_preview(img: Image.Image, template: Template) -> Image.Image:
    preview = img.copy()
    draw = ImageDraw.Draw(preview)
    
    if template.id_block:
        r = template.id_block
        draw.rectangle([r.x, r.y, r.x2, r.y2], outline="red", width=4)
        draw.text((r.x+10, r.y+10), "ID", fill="red")
    
    for i, block in enumerate(template.q_blocks, 1):
        r = block.rect
        draw.rectangle([r.x, r.y, r.x2, r.y2], outline="green", width=4)
        draw.text((r.x+10, r.y+10), f"Q{block.start_q}-{block.end_q}", fill="green")
    
    return preview


def main():
    st.set_page_config(page_title="OMR Scanner", layout="wide")
    
    st.title("✅ نظام تصحيح البابل شيت - تحديد بالماوس")
    st.markdown("---")
    
    # Session State
    if "template" not in st.session_state:
        st.session_state.template = None
    if "template_img" not in st.session_state:
        st.session_state.template_img = None
    
    # Layout
    col1, col2 = st.columns([1.5, 1])
    
    # ======================
    # RIGHT: Settings
    # ======================
    with col2:
        st.subheader("⚙️ الإعدادات")
        
        template_file = st.file_uploader("📄 النموذج", type=["pdf", "png", "jpg"])
        
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
                
                st.success(f"✅ {w}×{h}")
        
        if st.session_state.template_img:
            st.divider()
            
            col_a, col_b = st.columns(2)
            with col_a:
                choices = st.selectbox("الخيارات", [4, 5, 6], 0)
                st.session_state.template.num_choices = choices
            with col_b:
                id_digits = st.number_input("خانات الكود", 1, 12, 4, 1)
                st.session_state.template.id_digits = id_digits
            
            id_rows = st.number_input("صفوف الكود", 5, 15, 10, 1)
            st.session_state.template.id_rows = id_rows
            
            st.divider()
            
            mode = st.radio("التحديد", ["🆔 الكود", "📝 أسئلة"], 0)
            
            if mode == "📝 أسئلة":
                col_c, col_d, col_e = st.columns(3)
                with col_c:
                    start_q = st.number_input("من", 1, 500, 1)
                with col_d:
                    end_q = st.number_input("إلى", 1, 500, 20)
                with col_e:
                    num_rows = st.number_input("صفوف", 1, 200, 20)
            else:
                start_q = end_q = num_rows = 0
            
            st.info("🖱️ ارسم مستطيل على الصورة في اليسار")
            
            if st.session_state.template.q_blocks:
                st.divider()
                st.markdown("**البلوكات:**")
                for i, b in enumerate(st.session_state.template.q_blocks):
                    col_info, col_del = st.columns([3, 1])
                    with col_info:
                        st.text(f"{i+1}. Q{b.start_q}-{b.end_q}")
                    with col_del:
                        if st.button("🗑️", key=f"del{i}"):
                            st.session_state.template.q_blocks.pop(i)
                            st.rerun()
            
            st.divider()
            
            st.subheader("📂 الملفات")
            roster = st.file_uploader("📋 القائمة", type=["xlsx", "csv"])
            key_file = st.file_uploader("🔑 الإجابات", type=["pdf", "png", "jpg"])
            sheets = st.file_uploader("📚 الأوراق", type=["pdf", "png", "jpg"])
            
            strict = st.checkbox("وضع صارم", True)
    
    # ======================
    # LEFT: Canvas
    # ======================
    with col1:
        if st.session_state.template_img:
            st.subheader("🖱️ ارسم المستطيل بالماوس")
            
            # عرض معاينة أولاً
            preview = draw_preview(st.session_state.template_img, 
                                  st.session_state.template)
            
            # Canvas للرسم
            canvas_width = 800
            canvas_height = int(st.session_state.template_img.height * 
                              (canvas_width / st.session_state.template_img.width))
            
            st.markdown("**📌 اضغط واسحب لرسم مستطيل**")
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.1)",
                stroke_width=3,
                stroke_color="#FF0000" if mode == "🆔 الكود" else "#00FF00",
                background_image=preview,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                point_display_radius=0,
                key="canvas"
            )
            
            # معالجة الرسم
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                
                if objects:
                    # آخر مستطيل مرسوم
                    last_rect = objects[-1]
                    
                    # تحويل من حجم Canvas للحجم الأصلي
                    scale_x = st.session_state.template_img.width / canvas_width
                    scale_y = st.session_state.template_img.height / canvas_height
                    
                    x = int(last_rect["left"] * scale_x)
                    y = int(last_rect["top"] * scale_y)
                    w = int(last_rect["width"] * scale_x)
                    h = int(last_rect["height"] * scale_y)
                    
                    st.info(f"📏 المستطيل: ({x}, {y}) - العرض: {w}, الارتفاع: {h}")
                    
                    # زر الحفظ
                    if st.button("💾 حفظ المستطيل", type="primary", use_container_width=True):
                        if w < 10 or h < 10:
                            st.error("❌ المستطيل صغير جداً")
                        else:
                            rect = Rectangle(x, y, w, h)
                            
                            if mode == "🆔 الكود":
                                st.session_state.template.id_block = rect
                                st.success("✅ تم حفظ منطقة الكود")
                            else:
                                block = QuestionBlock(rect, start_q, end_q, num_rows)
                                st.session_state.template.q_blocks.append(block)
                                st.success(f"✅ بلوك {start_q}-{end_q}")
                            
                            st.rerun()
            
            st.divider()
            st.subheader("🚀 التصحيح")
            
            if st.button("▶️ ابدأ", type="primary", use_container_width=True):
                if not st.session_state.template.id_block:
                    st.error("❌ حدد منطقة الكود")
                    st.stop()
                
                if not st.session_state.template.q_blocks:
                    st.error("❌ أضف بلوك أسئلة")
                    st.stop()
                
                if not (roster and key_file and sheets):
                    st.error("❌ ارفع جميع الملفات")
                    st.stop()
                
                try:
                    # Load roster
                    if roster.name.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(roster)
                    else:
                        df = pd.read_csv(roster)
                    
                    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                    roster_dict = dict(zip(df["student_code"].astype(str).str.strip(),
                                         df["student_name"].astype(str).str.strip()))
                    
                    st.info(f"📋 {len(roster_dict)} طالب")
                    
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
                    
                    st.success(f"✅ {len(answer_key)} إجابة")
                    
                    # Grade
                    sheets_img = ImageProcessor.load_image(sheets.getvalue(), sheets.name)
                    sheets_bgr = cv2.cvtColor(np.array(sheets_img), cv2.COLOR_RGB2BGR)
                    
                    result = engine.grade_sheet(sheets_bgr, answer_key, roster_dict, strict)
                    
                    st.success("✅ اكتمل التصحيح!")
                    
                    df_results = pd.DataFrame([{
                        "الكود": result["id"],
                        "الاسم": result["name"],
                        "الصحيحة": result["score"],
                        "المجموع": result["total"],
                        "النسبة": f"{result['percentage']:.1f}%",
                        "الحالة": "ناجح ✓" if result["passed"] else "راسب ✗"
                    }])
                    
                    st.dataframe(df_results, use_container_width=True)
                    
                    buffer = io.BytesIO()
                    df_results.to_excel(buffer, index=False, engine='openpyxl')
                    
                    st.download_button(
                        "⬇️ تحميل Excel",
                        buffer.getvalue(),
                        "results.xlsx",
                        use_container_width=True
                    )
                
                except Exception as e:
                    st.error(f"❌ خطأ: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        else:
            st.info("📄 ارفع النموذج من اليمين")


if __name__ == "__main__":
    main()
