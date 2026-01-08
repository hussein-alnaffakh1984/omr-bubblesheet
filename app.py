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

def load_pages(file_bytes, filename, dpi=200):  # Lower DPI for speed
    """Load pages with memory management"""
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def bgr_to_bytes(bgr):
    _, buffer = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])  # Higher compression
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
    
    # Try prefix match (if code is longer)
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

# ==== MAIN APP ====
def main():
    st.set_page_config(page_title="🤖 AI OMR - Scalable", layout="wide")
    st.title("🤖 نظام OMR للأعداد الكبيرة")
    st.markdown("### 📊 500-700 طالب بدون مشاكل!")
    
    # Session state
    if 'answer_key' not in st.session_state: st.session_state.answer_key = {}
    if 'students' not in st.session_state: st.session_state.students = []
    if 'results' not in st.session_state: st.session_state.results = []
    if 'processed_pages' not in st.session_state: st.session_state.processed_pages = set()
    if 'duplicate_warnings' not in st.session_state: st.session_state.duplicate_warnings = []
    if 'allow_duplicates' not in st.session_state: st.session_state.allow_duplicates = False
    
    # Sidebar
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
        
        if st.button("🔄 Reset All", type="secondary"):
            st.session_state.answer_key = {}
            st.session_state.results = []
            st.session_state.processed_pages = set()
            st.rerun()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Answer Key", "2️⃣ Students", "3️⃣ Grade", "4️⃣ Results"])
    
    # TAB 1: Answer Key
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
            st.info(f"Loaded: {len(st.session_state.students)} students")
            with st.expander("View Students"):
                df = pd.DataFrame([{"ID": s.student_id, "Name": s.name, "Code": s.code} 
                                   for s in st.session_state.students[:50]])
                st.dataframe(df)
    
    # TAB 3: Grading
    with tab3:
        st.subheader("✅ Grading - Optimized for Large Scale")
        
        if not st.session_state.answer_key:
            st.warning("⚠️ Load Answer Key first")
            return
        if not st.session_state.students:
            st.warning("⚠️ Load Students first")
            return
        
        st.info("""
        💡 **للأعداد الكبيرة (500-700 طالب):**
        
        **الطريقة الموصى بها:**
        1. قسّم PDF الكبير لملفات أصغر (50-100 ورقة لكل ملف)
        2. ارفع ملف واحد في كل مرة
        3. عالج 20-30 ورقة في كل دفعة
        4. النتائج تتجمع تلقائياً
        
        **مثال:** 500 طالب
        - الملف 1: أوراق 1-100 (10 دفعات × 10 أوراق)
        - الملف 2: أوراق 101-200
        - إلخ...
        
        **الوقت المتوقع:** 5-7 ملفات × 5 دقائق = 30-35 دقيقة
        **التكلفة:** 500 × $0.003 = $1.50
        """)
        
        sheets = st.file_uploader(
            "ارفع ملف PDF (موصى به: 50-100 ورقة)",
            type=["pdf"],
            accept_multiple_files=False,
            key="sheets"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.slider("📦 Batch size", 5, 50, 20)
        with col2:
            auto_continue = st.checkbox("🔄 Auto-continue", value=True, help="استمر تلقائياً للدفعة التالية")
        
        st.markdown("---")
        st.subheader("🔍 إدارة التكرارات")
        
        dup_mode = st.radio(
            "كيف تتعامل مع الأكواد المكررة؟",
            options=[
                "⚠️ تحذير فقط (صحح الجميع)",
                "🚫 تجاهل التكرارات (صحح الأول فقط)",
                "✅ لا تفحص التكرارات (صحح كل شيء)"
            ],
            help="""
            **تحذير فقط:** يصحح كل الأوراق ويعطيك قائمة بالأكواد المكررة
            **تجاهل:** يصحح أول ورقة فقط ويتجاهل الباقي
            **لا تفحص:** يصحح كل الأوراق (قد يكون عندك طلاب بنفس الكود)
            """
        )
        
        if st.session_state.duplicate_warnings:
            st.warning(f"⚠️ تم اكتشاف {len(st.session_state.duplicate_warnings)} كود مكرر!")
            with st.expander("عرض الأكواد المكررة"):
                for dup in st.session_state.duplicate_warnings:
                    st.error(f"الكود {dup['code']} - الصفحات: {', '.join(map(str, dup['pages']))}")
        
        if sheets and 'current_file_pages' not in st.session_state:
            if st.button("🔍 Load File"):
                with st.spinner("Loading file..."):
                    b = read_bytes(sheets)
                    pages = load_pages(b, sheets.name, 200)
                    st.session_state.current_file_pages = pages
                    st.session_state.current_file_idx = 0
                    st.success(f"✅ Loaded {len(pages)} pages from {sheets.name}")
        
        if 'current_file_pages' in st.session_state:
            pages = st.session_state.current_file_pages
            current = st.session_state.current_file_idx
            total = len(pages)
            remaining = total - current
            
            st.metric("File Progress", f"{current}/{total} ({current/total*100:.0f}%)")
            
            if remaining > 0:
                if st.button(f"🚀 Process next {min(batch_size, remaining)}", type="primary") or auto_continue:
                    end = min(current + batch_size, total)
                    
                    progress = st.progress(0)
                    status = st.empty()
                    
                    processed_count = 0
                    
                    for i in range(current, end):
                        rel = i - current
                        status.text(f"Page {i+1}/{total} ({rel+1}/{end-current})")
                        progress.progress((rel+1)/(end-current))
                        
                        # Skip if already processed
                        if i in st.session_state.processed_pages:
                            status.text(f"⏭️ Page {i+1} already processed")
                            continue
                        
                        page = pages[i]
                        bgr = pil_to_bgr(page)
                        img = bgr_to_bytes(bgr)
                        
                        res = analyze_with_ai(img, api_key, False)
                        
                        if not res.success or not res.student_code:
                            st.warning(f"⚠️ Page {i+1}: Failed to read")
                            continue
                        
                        code = res.student_code.strip()
                        
                        # Strict validation
                        if not code.isdigit():
                            st.warning(f"⚠️ Page {i+1}: Bad code '{code}' (contains non-digits)")
                            continue
                        
                        if len(code) != 4:
                            st.warning(f"⚠️ Page {i+1}: Bad code '{code}' (must be exactly 4 digits, got {len(code)})")
                            continue
                        
                        code_int = int(code)
                        if code_int < 1000 or code_int > 1999:
                            st.warning(f"⚠️ Page {i+1}: Code {code} out of range (expected 1000-1999)")
                            continue
                        
                        student = find_student_by_code(st.session_state.students, code)
                        if not student:
                            st.warning(f"⚠️ Page {i+1}: Code {code} not found in student list")
                            continue
                        
                        # Check for duplicates based on mode
                        already_graded = any(r.detected_code == code for r in st.session_state.results)
                        
                        if already_graded:
                            if "تجاهل" in dup_mode:
                                # Mode 2: Skip duplicates
                                st.info(f"ℹ️ Page {i+1}: Code {code} ({student.name}) already graded - skipping")
                                st.session_state.processed_pages.add(i)
                                continue
                            elif "تحذير" in dup_mode:
                                # Mode 1: Warn but continue grading
                                st.warning(f"⚠️ Page {i+1}: Code {code} is DUPLICATE - grading anyway")
                                
                                # Track duplicate
                                existing_dup = next((d for d in st.session_state.duplicate_warnings if d['code'] == code), None)
                                if existing_dup:
                                    existing_dup['pages'].append(i+1)
                                else:
                                    st.session_state.duplicate_warnings.append({
                                        'code': code,
                                        'name': student.name,
                                        'pages': [i+1]
                                    })
                            # Mode 3: No check - continues automatically
                        
                        score, total_q = grade_student(res.answers, st.session_state.answer_key)
                        
                        st.session_state.results.append(GradingResult(
                            student.student_id, student.name, code, score, total_q, i+1
                        ))
                        
                        st.session_state.processed_pages.add(i)
                        processed_count += 1
                        
                        status.text(f"✅ Page {i+1}: {code} - {student.name} ({score}/{total_q})")
                        
                        # Free memory
                        del page, bgr, img
                    
                    st.session_state.current_file_idx = end
                    
                    # Force garbage collection
                    gc.collect()
                    
                    st.success(f"✅ Processed {processed_count} pages")
                    
                    if end >= total:
                        st.balloons()
                        st.success("🎉 File complete!")
                        del st.session_state.current_file_pages
                        del st.session_state.current_file_idx
                    elif auto_continue:
                        time.sleep(1)
                        st.rerun()
            else:
                st.success("File complete! Upload next file or go to Results.")
    
    # TAB 4: Results
    with tab4:
        st.subheader("📊 Results")
        
        if not st.session_state.results:
            st.info("No results yet")
            return
        
        # Duplicate warnings section
        if st.session_state.duplicate_warnings:
            st.error(f"⚠️ **تحذير: تم اكتشاف {len(st.session_state.duplicate_warnings)} كود مكرر!**")
            
            with st.expander("🔍 تفاصيل الأكواد المكررة", expanded=True):
                st.markdown("""
                **هذه الأكواد ظهرت في أكثر من ورقة:**
                - قد يكون طالب كتب كود زميله بالخطأ
                - راجع هذه الأوراق يدوياً
                - تحقق من الإجابات والخط
                """)
                
                for dup in st.session_state.duplicate_warnings:
                    st.warning(f"""
                    **الكود:** {dup['code']} - **الاسم:** {dup['name']}  
                    **ظهر في الصفحات:** {', '.join(map(str, dup['pages']))}  
                    **عدد التكرارات:** {len(dup['pages'])} مرة
                    """)
                
                # Show affected results
                dup_codes = [d['code'] for d in st.session_state.duplicate_warnings]
                dup_results = [r for r in st.session_state.results if r.detected_code in dup_codes]
                
                if dup_results:
                    st.markdown("**الأوراق المتأثرة:**")
                    dup_df = pd.DataFrame([{
                        "Page": r.page_number,
                        "Code": r.detected_code,
                        "Name": r.name,
                        "Score": r.score
                    } for r in dup_results])
                    st.dataframe(dup_df, width='stretch')
            
            st.markdown("---")
        
        scores = [r.score/r.total*100 for r in st.session_state.results]
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Graded", len(scores))
        with col2: st.metric("Average", f"{np.mean(scores):.1f}%")
        with col3: st.metric("Max", f"{np.max(scores):.1f}%")
        with col4: st.metric("Min", f"{np.min(scores):.1f}%")
        
        df = pd.DataFrame([{
            "Page": r.page_number,
            "ID": r.student_id, 
            "Name": r.name, 
            "Code": r.detected_code,
            "Score": r.score,
            "%": f"{r.score/r.total*100:.0f}"
        } for r in st.session_state.results])
        
        st.dataframe(df, width='stretch')
        
        # Duplicate cleaning options
        if st.session_state.duplicate_warnings:
            st.markdown("---")
            st.subheader("🧹 تنظيف التكرارات")
            
            clean_method = st.radio(
                "كيف تريد التعامل مع التكرارات؟",
                options=[
                    "احتفظ بالأول فقط",
                    "احتفظ بالأعلى درجة",
                    "احتفظ بالأقل درجة (للمراجعة)",
                    "احتفظ بالجميع (Excel سيظهر كلهم)"
                ]
            )
            
            if st.button("🧹 إنشاء نتائج نظيفة"):
                if "الأول" in clean_method:
                    # Keep first occurrence
                    clean_results = []
                    seen_codes = set()
                    for r in sorted(st.session_state.results, key=lambda x: x.page_number):
                        if r.detected_code not in seen_codes:
                            clean_results.append(r)
                            seen_codes.add(r.detected_code)
                    st.success(f"✅ تم! {len(clean_results)} نتيجة نظيفة (حذف {len(st.session_state.results) - len(clean_results)} تكرار)")
                    st.session_state.clean_results = clean_results
                
                elif "الأعلى" in clean_method:
                    # Keep highest score
                    from collections import defaultdict
                    by_code = defaultdict(list)
                    for r in st.session_state.results:
                        by_code[r.detected_code].append(r)
                    
                    clean_results = []
                    for code, results in by_code.items():
                        best = max(results, key=lambda x: x.score)
                        clean_results.append(best)
                    st.success(f"✅ تم! {len(clean_results)} نتيجة (أفضل درجة لكل كود)")
                    st.session_state.clean_results = clean_results
                
                elif "الأقل" in clean_method:
                    # Keep lowest score (for review)
                    from collections import defaultdict
                    by_code = defaultdict(list)
                    for r in st.session_state.results:
                        by_code[r.detected_code].append(r)
                    
                    clean_results = []
                    for code, results in by_code.items():
                        worst = min(results, key=lambda x: x.score)
                        clean_results.append(worst)
                    st.success(f"✅ تم! {len(clean_results)} نتيجة (أقل درجة للمراجعة)")
                    st.session_state.clean_results = clean_results
                
                else:
                    # Keep all
                    st.session_state.clean_results = st.session_state.results
        
        # Export buttons
        st.markdown("---")
        results_to_export = st.session_state.get('clean_results', st.session_state.results)
        
        if st.button("📥 Export Excel", type="primary"):
            excel = export_results(results_to_export)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            status_text = f"({len(results_to_export)} نتيجة"
            if 'clean_results' in st.session_state and len(results_to_export) < len(st.session_state.results):
                status_text += f" - تم تنظيف {len(st.session_state.results) - len(results_to_export)} تكرار"
            status_text += ")"
            
            st.download_button(
                f"⬇️ Download {status_text}", 
                excel, 
                f"results_{ts}.xlsx", 
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
