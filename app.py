"""
🤖 AI-Powered OMR - Complete System
- Answer key detection with AI
- Student registration from Excel
- Batch grading
- Results export
"""
import io
import base64
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
from datetime import datetime


# ==============================
# Helper functions
# ==============================
def read_bytes(uploaded_file) -> bytes:
    if uploaded_file is None:
        return b""
    try:
        return uploaded_file.getbuffer().tobytes()
    except Exception:
        try:
            return uploaded_file.read()
        except Exception:
            return b""


def load_pages(file_bytes: bytes, filename: str, dpi: int = 250) -> List[Image.Image]:
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def bgr_to_bytes(bgr: np.ndarray) -> bytes:
    _, buffer = cv2.imencode('.png', bgr)
    return buffer.tobytes()


@dataclass
class AIResult:
    answers: Dict[int, str]
    confidence: str
    notes: List[str]
    success: bool
    student_code: Optional[str] = None


@dataclass
class StudentRecord:
    student_id: str
    name: str
    code: str


@dataclass
class GradingResult:
    student_id: str
    name: str
    detected_code: str
    student_answers: Dict[int, str]
    score: int
    total: int
    percentage: float
    details: List[Dict]


# ==============================
# 🤖 AI Vision Analysis
# ==============================
def analyze_with_ai(image_bytes: bytes, api_key: str, is_answer_key: bool = True) -> AIResult:
    """
    Use Claude Vision API to analyze OMR sheet
    """
    if not api_key or len(api_key) < 20:
        return AIResult(
            answers={},
            confidence="no_api",
            notes=["❌ API Key مطلوب"],
            success=False
        )
    
    try:
        import anthropic
    except ImportError:
        return AIResult(
            answers={},
            confidence="error",
            notes=["❌ مكتبة anthropic غير مثبتة"],
            success=False
        )
    
    try:
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        if is_answer_key:
            prompt = """
أنت نظام OMR ذكي. انظر لورقة الإجابة النموذجية وحللها:

**مهمتك:**
1. احصي الفقاعات المظللة في كل سؤال
2. حدد الإجابة الصحيحة لكل سؤال (A, B, C, أو D)
3. تجاهل أرقام الأسئلة على اليسار
4. إذا كان هناك X على فقاعة، تجاهلها

**أعطني JSON فقط:**
```json
{
  "answers": {
    "1": "C",
    "2": "B",
    ...
  },
  "confidence": "high",
  "notes": []
}
```
"""
        else:
            prompt = """
أنت نظام OMR ذكي. انظر لورقة إجابة الطالب وحللها:

**مهمتك:**
1. **أولاً: اقرأ الكود (ID) من الفقاعات المظللة في الأعلى**
   - كل صف = رقم من الكود (0-9)
   - ظلل فقاعة واحدة في كل صف
   - الكود عادة 10 أرقام
   - القسم الأعلى من الورقة

2. **ثانياً: اقرأ الإجابات من الأسئلة**
   - كل سؤال له فقاعة واحدة مظللة (A, B, C, أو D)
   - تجاهل أرقام الأسئلة
   - القسم السفلي من الورقة

**أعطني JSON فقط:**
```json
{
  "student_code": "1234567890",
  "answers": {
    "1": "C",
    "2": "B",
    ...
  },
  "confidence": "high",
  "notes": []
}
```
"""
        
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }],
        )
        
        response_text = message.content[0].text
        
        import json
        import re
        
        json_text = response_text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].split("```")[0].strip()
        
        try:
            result = json.loads(json_text)
        except:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError("لم يتم العثور على JSON")
        
        answers = {int(k): v for k, v in result.get("answers", {}).items()}
        student_code = result.get("student_code", None) if not is_answer_key else None
        
        return AIResult(
            answers=answers,
            confidence=result.get("confidence", "medium"),
            notes=result.get("notes", []),
            success=True,
            student_code=student_code
        )
        
    except Exception as e:
        return AIResult(
            answers={},
            confidence="error",
            notes=[f"❌ خطأ: {str(e)}"],
            success=False
        )


# ==============================
# Student Management
# ==============================
def load_students_from_excel(file_bytes: bytes) -> List[StudentRecord]:
    """
    Load student records from Excel file
    Expected columns: student_id, name, code
    """
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
        
        # Try different column name variations
        id_col = None
        name_col = None
        code_col = None
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'id' in col_lower or 'رقم' in col_lower:
                id_col = col
            elif 'name' in col_lower or 'اسم' in col_lower:
                name_col = col
            elif 'code' in col_lower or 'كود' in col_lower or 'رمز' in col_lower:
                code_col = col
        
        if not all([id_col, name_col, code_col]):
            st.error("❌ يجب أن يحتوي الملف على أعمدة: ID, Name, Code")
            st.info(f"الأعمدة الموجودة: {', '.join(df.columns)}")
            return []
        
        students = []
        for _, row in df.iterrows():
            students.append(StudentRecord(
                student_id=str(row[id_col]),
                name=str(row[name_col]),
                code=str(row[code_col])
            ))
        
        return students
        
    except Exception as e:
        st.error(f"❌ خطأ في قراءة ملف Excel: {e}")
        return []


def find_student_by_code(students: List[StudentRecord], code: str) -> Optional[StudentRecord]:
    """Find student by code"""
    for student in students:
        if student.code == code:
            return student
    return None


# ==============================
# Grading
# ==============================
def grade_student(student_answers: Dict[int, str], answer_key: Dict[int, str]) -> Tuple[int, int, List[Dict]]:
    """
    Grade student answers against answer key
    Returns: (score, total, details)
    """
    details = []
    score = 0
    total = len(answer_key)
    
    for q_num in sorted(answer_key.keys()):
        correct_answer = answer_key[q_num]
        student_answer = student_answers.get(q_num, "?")
        
        is_correct = student_answer == correct_answer
        if is_correct:
            score += 1
        
        details.append({
            "Question": q_num,
            "Correct": correct_answer,
            "Student": student_answer,
            "Status": "✅" if is_correct else "❌"
        })
    
    return score, total, details


# ==============================
# Export Results
# ==============================
def export_results_to_excel(results: List[GradingResult]) -> bytes:
    """Export grading results to Excel"""
    # Summary sheet
    summary_data = []
    for result in results:
        summary_data.append({
            "Student ID": result.student_id,
            "Name": result.name,
            "Code": result.detected_code,
            "Score": result.score,
            "Total": result.total,
            "Percentage": f"{result.percentage:.1f}%",
            "Grade": get_grade(result.percentage)
        })
    
    # Detailed sheet
    detailed_data = []
    for result in results:
        for detail in result.details:
            detailed_data.append({
                "Student ID": result.student_id,
                "Name": result.name,
                "Question": detail["Question"],
                "Correct Answer": detail["Correct"],
                "Student Answer": detail["Student"],
                "Status": detail["Status"]
            })
    
    # Create Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        pd.DataFrame(detailed_data).to_excel(writer, sheet_name='Details', index=False)
    
    return output.getvalue()


def get_grade(percentage: float) -> str:
    """Convert percentage to grade"""
    if percentage >= 90:
        return "A"
    elif percentage >= 80:
        return "B"
    elif percentage >= 70:
        return "C"
    elif percentage >= 60:
        return "D"
    else:
        return "F"


# ==============================
# Main App
# ==============================
def main():
    st.set_page_config(
        page_title="🤖 AI OMR System",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 نظام تصحيح OMR الكامل بالذكاء الاصطناعي")
    st.markdown("### نظام متكامل: Answer Key + قائمة الطلاب + التصحيح + النتائج")
    
    # Initialize session state
    if 'answer_key' not in st.session_state:
        st.session_state.answer_key = {}
    if 'students' not in st.session_state:
        st.session_state.students = []
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # Sidebar - API Key
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        api_key = ""
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            if api_key:
                st.success("✅ API Key من Secrets")
        except:
            pass
        
        if not api_key:
            api_key = st.text_input(
                "🔑 API Key",
                type="password",
                placeholder="sk-ant-..."
            )
        
        st.markdown("---")
        st.metric("Answer Key", f"{len(st.session_state.answer_key)} أسئلة")
        st.metric("Students", f"{len(st.session_state.students)} طالب")
        st.metric("Graded", f"{len(st.session_state.results)} ورقة")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Answer Key",
        "2️⃣ قائمة الطلاب", 
        "3️⃣ التصحيح",
        "4️⃣ النتائج"
    ])
    
    # ============================================================
    # TAB 1: Answer Key
    # ============================================================
    with tab1:
        st.subheader("📝 ورقة الإجابة النموذجية")
        
        key_file = st.file_uploader(
            "ارفع ورقة الإجابة النموذجية",
            type=["pdf", "png", "jpg"],
            key="key"
        )
        
        if key_file:
            key_bytes = read_bytes(key_file)
            pages = load_pages(key_bytes, key_file.name, 250)
            
            if pages:
                bgr = pil_to_bgr(pages[0])
                st.image(bgr_to_rgb(bgr), width='stretch')
                
                if st.button("🤖 تحليل", type="primary"):
                    if not api_key:
                        st.error("❌ أدخل API Key")
                    else:
                        with st.spinner("⏳ جاري التحليل..."):
                            img_bytes = bgr_to_bytes(bgr)
                            result = analyze_with_ai(img_bytes, api_key, True)
                            
                            if result.success:
                                st.session_state.answer_key = result.answers
                                st.success(f"✅ {len(result.answers)} سؤال")
                                
                                ans = " | ".join([f"Q{q}: {a}" for q, a in sorted(result.answers.items())])
                                st.info(ans)
                            else:
                                st.error("❌ فشل")
                                for n in result.notes:
                                    st.warning(n)
        
        if st.session_state.answer_key:
            st.markdown("---")
            df = pd.DataFrame([
                {"Q": q, "Answer": a}
                for q, a in sorted(st.session_state.answer_key.items())
            ])
            st.dataframe(df, width='stretch')
    
    # ============================================================
    # TAB 2: Students
    # ============================================================
    with tab2:
        st.subheader("👥 قائمة الطلاب")
        
        st.info("**Excel يجب أن يحتوي على:** ID, Name, Code")
        
        excel = st.file_uploader("ارفع Excel", type=["xlsx", "xls"], key="excel")
        
        if excel and st.button("📊 تحميل"):
            students = load_students_from_excel(read_bytes(excel))
            if students:
                st.session_state.students = students
                st.success(f"✅ {len(students)} طالب")
        
        if st.session_state.students:
            df = pd.DataFrame([
                {"ID": s.student_id, "Name": s.name, "Code": s.code}
                for s in st.session_state.students[:20]
            ])
            st.dataframe(df, width='stretch')
            
            if len(st.session_state.students) > 20:
                st.info(f"عرض 20 من {len(st.session_state.students)}")
    
    # ============================================================
    # TAB 3: Grading
    # ============================================================
    with tab3:
        st.subheader("✅ التصحيح")
        
        if not st.session_state.answer_key:
            st.warning("⚠️ حمّل Answer Key أولاً")
            return
        
        if not st.session_state.students:
            st.warning("⚠️ حمّل قائمة الطلاب أولاً")
            return
        
        sheets = st.file_uploader(
            "ارفع أوراق الطلاب",
            type=["pdf", "png", "jpg"],
            accept_multiple_files=True,
            key="sheets"
        )
        
        if sheets and st.button("🚀 ابدأ", type="primary"):
            if not api_key:
                st.error("❌ أدخل API Key")
                return
            
            progress = st.progress(0)
            status = st.empty()
            
            results = []
            unmatched_codes = []  # Track codes that weren't found
            
            for idx, f in enumerate(sheets):
                status.text(f"📝 {idx+1}/{len(sheets)}")
                progress.progress((idx+1)/len(sheets))
                
                try:
                    b = read_bytes(f)
                    p = load_pages(b, f.name, 250)
                    if not p:
                        continue
                    
                    bgr = pil_to_bgr(p[0])
                    img = bgr_to_bytes(bgr)
                    
                    res = analyze_with_ai(img, api_key, False)
                    
                    if res.success and res.student_code:
                        st_code = res.student_code
                        st_ans = res.answers
                        
                        student = find_student_by_code(st.session_state.students, st_code)
                        
                        if student:
                            score, total, details = grade_student(st_ans, st.session_state.answer_key)
                            pct = (score/total*100) if total > 0 else 0
                            
                            results.append(GradingResult(
                                student_id=student.student_id,
                                name=student.name,
                                detected_code=st_code,
                                student_answers=st_ans,
                                score=score,
                                total=total,
                                percentage=pct,
                                details=details
                            ))
                            status.text(f"✅ {st_code}: {student.name}")
                        else:
                            unmatched_codes.append(st_code)
                            st.warning(f"⚠️ الكود {st_code} غير موجود في القائمة")
                    else:
                        st.error(f"❌ فشل قراءة {f.name}")
                
                except Exception as e:
                    st.error(f"❌ {f.name}: {e}")
            
            st.session_state.results = results
            
            # Summary
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ تم التصحيح", len(results))
            with col2:
                st.metric("⚠️ غير موجود", len(unmatched_codes))
            with col3:
                st.metric("📝 الإجمالي", len(sheets))
            
            if unmatched_codes:
                st.error("### ⚠️ أكواد غير موجودة في قائمة الطلاب:")
                
                # Show unmatched codes
                codes_text = ", ".join(unmatched_codes)
                st.code(codes_text)
                
                # Show available codes for comparison
                with st.expander("🔍 الأكواد المتاحة في قائمة الطلاب (أول 20)"):
                    available = [s.code for s in st.session_state.students[:20]]
                    st.code(", ".join(available))
                    if len(st.session_state.students) > 20:
                        st.info(f"عرض 20 من {len(st.session_state.students)} طالب")
                
                st.info("""
                **💡 حلول:**
                1. تأكد من أن الأكواد في ملف Excel صحيحة
                2. تأكد من عدم وجود مسافات زيادة
                3. تأكد من أن الطلاب ظللوا الأكواد بشكل صحيح
                4. حمّل ملف Excel محدّث يحتوي على هذه الأكواد
                """)
            
            if results:
                st.success(f"✅ تم تصحيح {len(results)} ورقة بنجاح!")
    
    # ============================================================
    # TAB 4: Results
    # ============================================================
    with tab4:
        st.subheader("📊 النتائج")
        
        if not st.session_state.results:
            st.info("لا توجد نتائج")
            return
        
        # Stats
        scores = [r.percentage for r in st.session_state.results]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("الطلاب", len(scores))
        with col2:
            st.metric("المتوسط", f"{np.mean(scores):.1f}%")
        with col3:
            st.metric("الأعلى", f"{np.max(scores):.1f}%")
        with col4:
            st.metric("الأدنى", f"{np.min(scores):.1f}%")
        
        # Table
        st.markdown("---")
        df = pd.DataFrame([
            {
                "ID": r.student_id,
                "Name": r.name,
                "Code": r.detected_code,
                "Score": f"{r.score}/{r.total}",
                "%": f"{r.percentage:.1f}",
                "Grade": get_grade(r.percentage)
            }
            for r in st.session_state.results
        ])
        st.dataframe(df, width='stretch')
        
        # Export
        st.markdown("---")
        if st.button("📥 تصدير Excel", type="primary"):
            excel = export_results_to_excel(st.session_state.results)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                "⬇️ تحميل",
                excel,
                f"results_{ts}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()
