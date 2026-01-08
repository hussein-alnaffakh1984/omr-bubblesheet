"""
🤖 AI OMR - Batch Processing Version (No Timeout!)
معالجة على دفعات لتجنب التوقف
"""
import io, base64, time
from dataclasses import dataclass
from typing import Dict, List, Optional
import cv2, numpy as np, pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
from datetime import datetime

# Same helper functions as before...
def read_bytes(f):
    if not f: return b""
    try: return f.getbuffer().tobytes()
    except: 
        try: return f.read()
        except: return b""

def load_pages(file_bytes, filename, dpi=250):
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def bgr_to_bytes(bgr):
    _, buffer = cv2.imencode('.png', bgr)
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
    student_answers: Dict
    score: int
    total: int
    percentage: float
    details: List

def analyze_with_ai(image_bytes, api_key, is_answer_key=True):
    """AI Analysis - same as before"""
    if not api_key or len(api_key) < 20:
        return AIResult({}, "no_api", ["API Key required"], False)
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        if is_answer_key:
            prompt = "أنت OMR خبير. اقرأ ورقة Answer Key واعطني JSON: {\"answers\": {\"1\": \"C\", ...}, \"confidence\": \"high\"}"
        else:
            prompt = """اقرأ ورقة الطالب واعطني JSON:
{"student_code": "1013", "answers": {"1": "C", ...}, "confidence": "high"}

⚠️ **قواعد قراءة الإجابات (بالترتيب):**

**القاعدة 1 - X يلغي الفقاعة (أولوية قصوى!):**
```
Q1: [●X] A [●] B [ ] C [ ] D
     ملغية  صحيح
→ A عليها X = ملغية تماماً!
→ الإجابة: B ✅
```

**القاعدة 2 - فقاعة واحدة مظللة بدون X:**
```
Q2: [ ] A [●] B [ ] C [ ] D
→ الإجابة: B ✅
```

**القاعدة 3 - أكثر من فقاعة بدون X (خطأ طالب):**
```
Q3: [●●] A [●] B [ ] C [ ] D
     أكثر   أقل
     قتامة  قتامة
→ قارن القتامة
→ الإجابة: A (الأكثر قتامة)
→ Note: "Q3: multiple marks - selected darkest"
```

**القاعدة 4 - لا فقاعة مظللة:**
```
Q4: [ ] A [ ] B [ ] C [ ] D
→ الإجابة: "?"
```

**⚠️ أمثلة حرجة للتأكد:**

```
مثال 1:
Q5: [●X] A [●] B [●X] C [ ] D
     ملغ    ✓   ملغ
→ A ملغية (X)
→ C ملغية (X)
→ الإجابة: B ✅

مثال 2:
Q6: [●X] A [●●] B [●] C [ ] D
     ملغ    أكثر   أقل
→ A ملغية (X)
→ بين B و C: B أكثر قتامة
→ الإجابة: B ✅

مثال 3:
Q7: [X] A [●] B [●] C [ ] D
    ملغ   ✓    ✓
→ A ملغية (حتى لو غير مظللة)
→ بين B و C: نفس القتامة
→ الإجابة: B (الأولى) ✅
```

**⚠️ خوارزمية المعالجة:**
1. اقرأ كل فقاعات السؤال
2. احذف أي فقاعة عليها X (مظللة أو لا!)
3. من الفقاعات المتبقية (بدون X):
   - واحدة مظللة → هذه الإجابة
   - أكثر من واحدة → اختر الأكثر قتامة
   - لا شيء → "?"

**الكود: فقط الأرقام المظللة (4-10 رقم)**"""
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
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
    
    # Try prefix match
    if len(code_norm) > 4:
        for length in [4, 5, 6, 7, 8]:
            if len(code_norm) >= length:
                prefix = code_norm[:length]
                for s in students:
                    s_code = str(s.code).strip().replace(" ", "").replace("-", "")
                    if s_code == prefix: return s
    return None

def grade_student(student_answers, answer_key):
    """Grade student"""
    details, score = [], 0
    total = len(answer_key)
    for q in sorted(answer_key.keys()):
        correct = answer_key[q]
        student = student_answers.get(q, "?")
        is_correct = student == correct
        if is_correct: score += 1
        details.append({"Question": q, "Correct": correct, "Student": student, "Status": "✅" if is_correct else "❌"})
    return score, total, details

def export_results(results):
    """Export to Excel"""
    summary = [{"ID": r.student_id, "Name": r.name, "Code": r.detected_code, "Score": f"{r.score}/{r.total}", "%": f"{r.percentage:.1f}"} for r in results]
    detailed = []
    for r in results:
        for d in r.details:
            detailed.append({"ID": r.student_id, "Name": r.name, "Q": d["Question"], "Correct": d["Correct"], "Student": d["Student"], "Status": d["Status"]})
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)
        pd.DataFrame(detailed).to_excel(writer, sheet_name='Details', index=False)
    return output.getvalue()

# ==== MAIN APP ====
def main():
    st.set_page_config(page_title="🤖 AI OMR (Batch)", layout="wide")
    st.title("🤖 نظام OMR الذكي - معالجة على دفعات")
    st.markdown("### ⚡ لا توقف! معالجة تدريجية")
    
    # Session state
    if 'answer_key' not in st.session_state: st.session_state.answer_key = {}
    if 'students' not in st.session_state: st.session_state.students = []
    if 'results' not in st.session_state: st.session_state.results = []
    if 'pages_data' not in st.session_state: st.session_state.pages_data = []
    if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        api_key = ""
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            if api_key: st.success("✅ API Key")
        except: pass
        if not api_key:
            api_key = st.text_input("🔑 API Key", type="password", placeholder="sk-ant-...")
        
        st.markdown("---")
        st.metric("Answer Key", f"{len(st.session_state.answer_key)} Q")
        st.metric("Students", len(st.session_state.students))
        st.metric("Graded", len(st.session_state.results))
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Answer Key", "2️⃣ Students", "3️⃣ Grade", "4️⃣ Results"])
    
    # TAB 1: Answer Key
    with tab1:
        st.subheader("📝 Answer Key")
        key_file = st.file_uploader("Upload", type=["pdf","png","jpg"], key="key")
        if key_file:
            key_bytes = read_bytes(key_file)
            pages = load_pages(key_bytes, key_file.name, 250)
            if pages:
                st.image(cv2.cvtColor(pil_to_bgr(pages[0]), cv2.COLOR_BGR2RGB), width='stretch')
                if st.button("🤖 Analyze", type="primary"):
                    if not api_key: st.error("Need API Key")
                    else:
                        with st.spinner("Analyzing..."):
                            img = bgr_to_bytes(pil_to_bgr(pages[0]))
                            res = analyze_with_ai(img, api_key, True)
                            if res.success:
                                st.session_state.answer_key = res.answers
                                st.success(f"✅ {len(res.answers)} questions")
                                st.info(" | ".join([f"Q{q}: {a}" for q, a in sorted(res.answers.items())]))
                            else: st.error("Failed")
        if st.session_state.answer_key:
            df = pd.DataFrame([{"Q": q, "A": a} for q, a in sorted(st.session_state.answer_key.items())])
            st.dataframe(df, width='stretch')
    
    # TAB 2: Students
    with tab2:
        st.subheader("👥 Students")
        excel = st.file_uploader("Upload Excel (ID, Name, Code)", type=["xlsx","xls"], key="excel")
        if excel and st.button("📊 Load"):
            students = load_students_from_excel(read_bytes(excel))
            if students:
                st.session_state.students = students
                st.success(f"✅ {len(students)} students")
        if st.session_state.students:
            df = pd.DataFrame([{"ID": s.student_id, "Name": s.name, "Code": s.code} for s in st.session_state.students])
            st.dataframe(df, width='stretch')
    
    # TAB 3: Grading
    with tab3:
        st.subheader("✅ التصحيح التدريجي")
        
        if not st.session_state.answer_key:
            st.warning("⚠️ Load Answer Key first")
            return
        if not st.session_state.students:
            st.warning("⚠️ Load Students first")
            return
        
        sheets = st.file_uploader("Upload papers", type=["pdf","png","jpg"], accept_multiple_files=True, key="sheets")
        
        batch_size = st.select_slider("📦 Batch size", options=[5,10,15,20], value=10)
        
        col1, col2 = st.columns(2)
        with col1:
            skip_duplicates = st.checkbox("🚫 Skip duplicates", value=True, help="تجاهل الأكواد المكررة")
        with col2:
            fast_mode = st.checkbox("⚡ Fast mode", value=False, help="أسرع لكن قد يتعب النظام")
        
        if sheets and not st.session_state.pages_data:
            if st.button("🔍 Prepare files"):
                with st.spinner("Loading files..."):
                    for f in sheets:
                        b = read_bytes(f)
                        pages = load_pages(b, f.name, 250)
                        for p in pages:
                            st.session_state.pages_data.append((f.name, p))
                st.success(f"✅ Loaded {len(st.session_state.pages_data)} pages")
                st.session_state.current_idx = 0
        
        if st.session_state.pages_data:
            total = len(st.session_state.pages_data)
            current = st.session_state.current_idx
            remaining = total - current
            
            st.info(f"📊 Progress: {current}/{total} ({current/total*100:.1f}%) | Remaining: {remaining}")
            
            if remaining > 0:
                if st.button(f"🚀 Process next {min(batch_size, remaining)} pages", type="primary"):
                    if not api_key:
                        st.error("Need API Key")
                        return
                    
                    end = min(current + batch_size, total)
                    progress = st.progress(0)
                    status = st.empty()
                    
                    for i in range(current, end):
                        rel = i - current
                        status.text(f"Processing page {i+1}/{total} ({rel+1}/{end-current} in batch)")
                        progress.progress((rel+1)/(end-current))
                        
                        fname, page = st.session_state.pages_data[i]
                        bgr = pil_to_bgr(page)
                        img = bgr_to_bytes(bgr)
                        
                        # Conditional delay based on mode
                        if not fast_mode:
                            time.sleep(0.2)
                        
                        res = analyze_with_ai(img, api_key, False)
                        if not res.success or not res.student_code:
                            st.warning(f"⚠️ Page {i+1}: Failed")
                            continue
                        
                        code = res.student_code.strip()
                        if not code.isdigit() or len(code) < 4:
                            st.warning(f"⚠️ Page {i+1}: Bad code '{code}'")
                            continue
                        
                        student = find_student_by_code(st.session_state.students, code)
                        if not student:
                            st.warning(f"⚠️ Page {i+1}: Code {code} not found")
                            continue
                        
                        # Check for duplicates (if enabled)
                        if skip_duplicates:
                            already_graded = any(r.detected_code == code for r in st.session_state.results)
                            if already_graded:
                                st.info(f"ℹ️ Page {i+1}: Code {code} ({student.name}) already graded - skipping")
                                continue
                        
                        score, tot, details = grade_student(res.answers, st.session_state.answer_key)
                        pct = (score/tot*100) if tot > 0 else 0
                        
                        st.session_state.results.append(GradingResult(
                            student.student_id, student.name, code, res.answers, score, tot, pct, details
                        ))
                        
                        status.text(f"✅ Page {i+1}: {code} - {student.name} ({score}/{tot})")
                    
                    st.session_state.current_idx = end
                    st.success(f"✅ Batch complete! Processed {end-current} pages")
                    st.balloons()
                    
                    if end >= total:
                        st.success("🎉 ALL DONE!")
            else:
                st.success("🎉 All pages processed!")
                if st.button("🔄 Reset"):
                    st.session_state.pages_data = []
                    st.session_state.current_idx = 0
    
    # TAB 4: Results
    with tab4:
        st.subheader("📊 Results")
        
        if not st.session_state.results:
            st.info("No results yet")
            return
        
        scores = [r.percentage for r in st.session_state.results]
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Students", len(scores))
        with col2: st.metric("Average", f"{np.mean(scores):.1f}%")
        with col3: st.metric("Max", f"{np.max(scores):.1f}%")
        
        df = pd.DataFrame([{
            "ID": r.student_id, "Name": r.name, "Code": r.detected_code,
            "Score": f"{r.score}/{r.total}", "%": f"{r.percentage:.1f}"
        } for r in st.session_state.results])
        st.dataframe(df, width='stretch')
        
        if st.button("📥 Export Excel", type="primary"):
            excel = export_results(st.session_state.results)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("⬇️ Download", excel, f"results_{ts}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
