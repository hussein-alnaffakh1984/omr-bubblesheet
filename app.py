"""
🤖 AI OMR - Scalable Version for Large Classes (500-700 students)
معالجة قابلة للتوسع للأعداد الكبيرة
"""
import io, base64, time, gc
from dataclasses import dataclass
from typing import Dict, List, Optional
import cv2, numpy as np, pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
from datetime import datetime

# Same helper functions...
def read_bytes(f):
    if not f: return b""
    try: return f.getbuffer().tobytes()
    except: 
        try: return f.read()
        except: return b""

def load_pages(file_bytes, filename, dpi=150):
    """Load pages with aggressive memory management"""
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi, fmt='jpeg', jpegopt={'quality': 85, 'optimize': True})
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def bgr_to_bytes(bgr):
    _, buffer = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return buffer.tobytes()

@dataclass
class AIResult:
    answers: Dict
    confidence: str
    notes: List
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
    score: int
    total: int
    page_number: int = 0

def analyze_with_ai(image_bytes, api_key, is_answer_key=True):
    """AI Analysis - optimized"""
    if not api_key or len(api_key) < 20:
        return AIResult({}, "no_api", ["API Key required"], False)
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        if is_answer_key:
            prompt = "اقرأ Answer Key. JSON فقط: {\"answers\": {\"1\": \"C\", ...}}"
        else:
            prompt = """أنت نظام OMR خبير. اقرأ ورقة الطالب بدقة.

━━━━━━━━━━━━━━━━━━━━━━
📋 الكود (4 أرقام بالضبط)
━━━━━━━━━━━━━━━━━━━━━━

الكود في أعلى الورقة - شبكة أرقام.
اقرأ **4 صفوف فقط** - كل صف = رقم واحد.
النطاق الصحيح: **1000-1057**

مثال صحيح:
الصف 1: "1" → 1
الصف 2: "0" → 0
الصف 3: "1" → 1
الصف 4: "3" → 3
الكود = "1013" ✅

❌ تجنب:
- أكثر من 4 أرقام
- أقل من 4 أرقام
- أكواد خارج النطاق 1000-1057

━━━━━━━━━━━━━━━━━━━━━━
📋 الإجابات
━━━━━━━━━━━━━━━━━━━━━━

**القاعدة 1 - X يلغي الفقاعة (أولوية قصوى!):**
Q1: [●X] A [●] B [ ] C [ ] D
     ملغ    ✓
→ احذف A (عليها X)
→ الإجابة: B ✅

**القاعدة 2 - فقاعة واحدة:**
Q2: [ ] A [●] B [ ] C [ ] D
→ الإجابة: B ✅

**القاعدة 3 - أكثر من فقاعة:**
Q3: [●●] A [●] B [ ] C [ ] D
     أكثر   أقل
     قتامة  قتامة
→ الإجابة: A (الأكثر قتامة) ✅

**خوارزمية:**
1. احذف أي فقاعة عليها X
2. من المتبقي: اختر الأكثر قتامة
3. إذا لا شيء: "?"

JSON فقط:
{"student_code": "1013", "answers": {"1": "C", "2": "B", ...}}"""
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        
        response_text = message.content[0].text
        
        import json, re
        json_text = response_text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].split("```")[0].strip()
        
        try:
            result = json.loads(json_text)
        except:
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match: result = json.loads(match.group())
            else: raise ValueError("No JSON")
        
        answers = {int(k): v for k, v in result.get("answers", {}).items()}
        student_code = result.get("student_code") if not is_answer_key else None
        
        return AIResult(answers, result.get("confidence", "medium"), result.get("notes", []), True, student_code)
    
    except Exception as e:
        return AIResult({}, "error", [str(e)], False)

def load_students_from_excel(file_bytes):
    """Load students from Excel"""
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
        id_col = name_col = code_col = None
        for col in df.columns:
            cl = str(col).lower().strip()
            if 'id' in cl or 'رقم' in cl: id_col = col
            elif 'name' in cl or 'اسم' in cl: name_col = col
            elif 'code' in cl or 'كود' in cl or 'رمز' in cl: code_col = col
        
        if not all([id_col, name_col, code_col]):
            return []
        
        students = []
        for _, row in df.iterrows():
            students.append(StudentRecord(str(row[id_col]), str(row[name_col]), str(row[code_col])))
        return students
    except Exception as e:
        st.error(f"Excel error: {e}")
        return []

def find_student_by_code(students, code):
    """Find student with flexible matching"""
    code_norm = str(code).strip().replace(" ", "").replace("-", "")
    for s in students:
        s_code = str(s.code).strip().replace(" ", "").replace("-", "")
        if s_code == code_norm: return s
    
    if len(code_norm) > 4:
        for length in [4, 5, 6]:
            if len(code_norm) >= length:
                prefix = code_norm[:length]
                for s in students:
                    s_code = str(s.code).strip().replace(" ", "").replace("-", "")
                    if s_code == prefix: return s
    return None

def grade_student(student_answers, answer_key):
    """Grade student"""
    score = sum(1 for q in answer_key.keys() if student_answers.get(q) == answer_key[q])
    return score, len(answer_key)

def export_results(results):
    """Export to Excel - minimal format"""
    data = [{
        "Page": r.page_number,
        "ID": r.student_id, 
        "Name": r.name, 
        "Code": r.detected_code, 
        "Score": r.score
    } for r in results]
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(data).to_excel(writer, sheet_name='Results', index=False)
    return output.getvalue()

def main():
    st.set_page_config(page_title="🤖 AI OMR", layout="wide")
    st.title("🤖 نظام OMR للأعداد الكبيرة")
    st.markdown("### 📊 500-700 طالب بدون مشاكل!")
    
    if 'answer_key' not in st.session_state: st.session_state.answer_key = {}
    if 'students' not in st.session_state: st.session_state.students = []
    if 'results' not in st.session_state: st.session_state.results = []
    if 'processed_pages' not in st.session_state: st.session_state.processed_pages = set()
    
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        api_key = ""
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            if api_key: st.success("✅ API Key")
        except: pass
        if not api_key:
            api_key = st.text_input("🔑 API Key", type="password")
        
        st.markdown("---")
        st.metric("Answer Key", f"{len(st.session_state.answer_key)} Q")
        st.metric("Students", len(st.session_state.students))
        st.metric("Graded", len(st.session_state.results))
        
        if st.session_state.results:
            avg = np.mean([r.score/r.total*100 for r in st.session_state.results])
            st.metric("Average", f"{avg:.1f}%")
        
        if st.button("🔄 Reset All"):
            st.session_state.answer_key = {}
            st.session_state.results = []
            st.session_state.processed_pages = set()
            st.rerun()
    
    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Answer Key", "2️⃣ Students", "3️⃣ Grade", "4️⃣ Results"])
    
    with tab1:
        st.subheader("📝 Answer Key")
        key_file = st.file_uploader("Upload Answer Key", type=["pdf","png","jpg"], key="key")
        if key_file:
            if st.button("🤖 Analyze", type="primary"):
                if not api_key: 
                    st.error("❌ Need API Key")
                else:
                    with st.spinner("Analyzing..."):
                        b = read_bytes(key_file)
                        pages = load_pages(b, key_file.name, 200)
                        if pages:
                            img = bgr_to_bytes(pil_to_bgr(pages[0]))
                            res = analyze_with_ai(img, api_key, True)
                            if res.success:
                                st.session_state.answer_key = res.answers
                                st.success(f"✅ {len(res.answers)} questions")
                            else: st.error("Failed")
        
        if st.session_state.answer_key:
            st.info(" | ".join([f"Q{q}: {a}" for q, a in sorted(st.session_state.answer_key.items())]))
    
    with tab2:
        st.subheader("👥 Students")
        excel = st.file_uploader("Upload Excel (ID, Name, Code)", type=["xlsx","xls"], key="excel")
        if excel and st.button("📊 Load"):
            students = load_students_from_excel(read_bytes(excel))
            if students:
                st.session_state.students = students
                st.success(f"✅ {len(students)} students")
        
        if st.session_state.students:
            st.info(f"Loaded: {len(st.session_state.students)} students")
    
    with tab3:
        st.subheader("✅ Grading")
        
        if not st.session_state.answer_key:
            st.warning("⚠️ Load Answer Key first")
            return
        if not st.session_state.students:
            st.warning("⚠️ Load Students first")
            return
        
        sheets = st.file_uploader("Upload PDF (⚠️ Max 50 pages)", type=["pdf"], key="sheets")
        
        batch_size = st.slider("📦 Batch size", 5, 20, 10)
        
        if sheets and 'current_file_pages' not in st.session_state:
            if st.button("🔍 Load File"):
                with st.spinner("Loading..."):
                    b = read_bytes(sheets)
                    pages = load_pages(b, sheets.name, 200)
                    st.session_state.current_file_pages = pages
                    st.session_state.current_file_idx = 0
                    st.success(f"✅ {len(pages)} pages")
        
        if 'current_file_pages' in st.session_state:
            pages = st.session_state.current_file_pages
            current = st.session_state.current_file_idx
            total = len(pages)
            
            if current < total:
                if st.button(f"🚀 Process {min(batch_size, total-current)}", type="primary"):
                    end = min(current + batch_size, total)
                    
                    for i in range(current, end):
                        if i in st.session_state.processed_pages:
                            continue
                        
                        page = pages[i]
                        bgr = pil_to_bgr(page)
                        img = bgr_to_bytes(bgr)
                        
                        res = analyze_with_ai(img, api_key, False)
                        
                        if not res.success or not res.student_code:
                            st.warning(f"⚠️ Page {i+1}: Failed")
                            continue
                        
                        code = res.student_code.strip()
                        
                        if not code.isdigit() or len(code) != 4:
                            st.warning(f"⚠️ Page {i+1}: Bad code '{code}'")
                            continue
                        
                        student = find_student_by_code(st.session_state.students, code)
                        if not student:
                            st.warning(f"⚠️ Page {i+1}: Code {code} not found")
                            continue
                        
                        score, total_q = grade_student(res.answers, st.session_state.answer_key)
                        
                        st.session_state.results.append(GradingResult(
                            student.student_id, student.name, code, score, total_q, i+1
                        ))
                        
                        st.session_state.processed_pages.add(i)
                        st.success(f"✅ Page {i+1}: {code} - {student.name} ({score}/{total_q})")
                    
                    st.session_state.current_file_idx = end
                    
                    if end >= total:
                        del st.session_state.current_file_pages
                        del st.session_state.current_file_idx
                        gc.collect()
    
    with tab4:
        st.subheader("📊 Results")
        
        if not st.session_state.results:
            st.info("No results yet")
            return
        
        df = pd.DataFrame([{
            "Page": r.page_number,
            "ID": r.student_id, 
            "Name": r.name, 
            "Code": r.detected_code,
            "Score": r.score
        } for r in st.session_state.results])
        
        st.dataframe(df)
        
        if st.button("📥 Export Excel", type="primary"):
            excel = export_results(st.session_state.results)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("⬇️ Download", excel, f"results_{ts}.xlsx")

if __name__ == "__main__":
    main()
