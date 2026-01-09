"""
🤖 AI OMR - Scalable Version for Large Classes (500-700 students)
معالجة قابلة للتوسع للأعداد الكبيرة
"""
import io, base64, time, gc, re
from dataclasses import dataclass
from typing import Dict, List, Optional
import cv2, numpy as np, pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
from datetime import datetime

# OCR for code extraction
try:
    import pytesseract
    HAS_TESSERACT = True
except:
    HAS_TESSERACT = False

# Same helper functions...
def read_bytes(f):
    if not f: return b""
    try: return f.getbuffer().tobytes()
    except: 
        try: return f.read()
        except: return b""

def load_pages(file_bytes, filename, dpi=150):  # Lower DPI: 150 instead of 200
    """Load pages with aggressive memory management"""
    if filename.lower().endswith(".pdf"):
        # Process in smaller chunks
        pages = convert_from_bytes(file_bytes, dpi=dpi, fmt='jpeg', jpegopt={'quality': 85, 'optimize': True})
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def bgr_to_bytes(bgr):
    _, buffer = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])  # Higher compression
    return buffer.tobytes()

# OCR-based code extraction
def extract_code_with_ocr(bgr_image):
    """Extract 4-digit code using OCR (more accurate for numbers)"""
    if not HAS_TESSERACT:
        return None, 0
    
    try:
        h, w = bgr_image.shape[:2]
        
        # ROI for code area (top-left section)
        y1, y2 = int(0.145 * h), int(0.285 * h)
        x1, x2 = int(0.080 * w), int(0.440 * w)
        roi = bgr_image[y1:y2, x1:x2].copy()
        
        # Preprocess
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        
        # Try multiple thresholds
        variants = []
        
        # Otsu inverse
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants.append(th1)
        
        # Adaptive mean
        th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 7)
        variants.append(th2)
        
        # Adaptive gaussian
        th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 7)
        variants.append(th3)
        
        best_code = None
        best_score = -999
        
        for variant in variants:
            # Upscale for better OCR
            big = cv2.resize(variant, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            
            # OCR with digit whitelist
            config = "--psm 7 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(big, config=config).strip()
            
            # Extract 4-digit codes
            codes = re.findall(r'\b(1[0-9]{3})\b', text)
            
            if codes:
                code = codes[0]
                code_int = int(code)
                
                # Score based on validity
                score = 50  # base score
                if 1000 <= code_int <= 1999:
                    score += 50
                
                if score > best_score:
                    best_code = code
                    best_score = score
        
        return best_code, best_score
    
    except Exception as e:
        return None, 0

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
            prompt = """Read the ANSWER KEY sheet carefully.

This is the CORRECT ANSWERS sheet (not a student sheet).
It shows the right answer for each question.

There are 10 questions, each with 4 choices: A, B, C, D
One bubble is filled for each question - that's the correct answer.

Read the filled bubble for each question (1-10).

RESPOND WITH JSON ONLY:
{
  "answers": {
    "1": "C",
    "2": "B",
    "3": "A",
    "4": "D",
    "5": "A",
    "6": "C",
    "7": "B",
    "8": "D",
    "9": "A",
    "10": "B"
  }
}"""
        else:
            prompt = """You are an expert OMR (Optical Mark Recognition) system. Read this student answer sheet with EXTREME precision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 STUDENT CODE GRID (TOP OF PAGE) - READ WITH EXTREME CARE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The code grid has FOUR VERTICAL COLUMNS (reading LEFT to RIGHT):

COLUMN 1 (First digit):    ⓪ ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨
COLUMN 2 (Second digit):   ⓪ ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨
COLUMN 3 (Third digit):    ⓪ ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨
COLUMN 4 (Fourth digit):   ⓪ ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨

CRITICAL INSTRUCTIONS:
1. Look at EACH COLUMN separately - treat each like an independent question
2. Find the FILLED/DARKEST bubble in each column
3. The code MUST start with "1" (first column = 1)
4. Valid range: 1000-1057
5. Output EXACTLY 4 digits - no more, no less

COMMON MISTAKES TO AVOID:
❌ Confusing 0 ↔ 8 (zero vs eight)
❌ Confusing 1 ↔ 7 (one vs seven)  
❌ Confusing 3 ↔ 8 (three vs eight)
❌ Confusing 5 ↔ 6 (five vs six)
❌ Confusing 7 ↔ 9 (seven vs nine)
❌ Reading wrong column order

STEP-BY-STEP PROCESS:
Step 1: Locate the code grid (top-left area of page)
Step 2: Read Column 1 → Find filled bubble → Usually "1"
Step 3: Read Column 2 → Find filled bubble → 0-9
Step 4: Read Column 3 → Find filled bubble → 0-9
Step 5: Read Column 4 → Find filled bubble → 0-9
Step 6: Combine: [digit1][digit2][digit3][digit4]

EXAMPLE:
Column 1: Bubble ① is filled → "1"
Column 2: Bubble ⓪ is filled → "0"
Column 3: Bubble ① is filled → "1"
Column 4: Bubble ⑦ is filled → "7"
Final code = "1017" ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 ANSWERS (10 Questions: A, B, C, D)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 - X mark CANCELS a bubble (HIGHEST PRIORITY):
Q1: [●X] A  [●] B  [ ] C  [ ] D
    ^^^^    ^^^
  CANCEL   ANSWER
→ Ignore A (has X mark)
→ Answer: B ✅

RULE 2 - Single filled bubble:
Q2: [ ] A  [●] B  [ ] C  [ ] D
→ Answer: B ✅

RULE 3 - Multiple filled bubbles (NO X marks):
Q3: [●●] A  [●] B  [ ] C  [ ] D
    ^^^^    ^^^
   DARKER  LIGHTER
→ Answer: A (darkest) ✅

ALGORITHM:
1. Remove any bubble with X mark
2. From remaining: choose DARKEST
3. If none: "?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESPOND WITH JSON ONLY (no extra text):
{
  "col1": "1",
  "col2": "0",
  "col3": "1",
  "col4": "7",
  "student_code": "1017",
  "answers": {
    "1": "C",
    "2": "B",
    "3": "A",
    "4": "D",
    "5": "A",
    "6": "C",
    "7": "B",
    "8": "D",
    "9": "A",
    "10": "B"
  }
}"""
        
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
        1. قسّم PDF الكبير لملفات أصغر (**30-50 ورقة لكل ملف** - مهم!)
        2. ارفع ملف واحد في كل مرة
        3. عالج 10-20 ورقة في كل دفعة
        4. النتائج تتجمع تلقائياً
        5. استخدم AI لقراءة الأكواد والإجابات بدقة
        
        ⚠️ **لتجنب Memory Error:**
        - لا ترفع ملفات أكبر من 50 صفحة
        - استخدم batch size صغير (10-20)
        - لو ظهر خطأ memory: اضغط "Reboot" وأعد المحاولة بملفات أصغر
        
        **مثال:** 500 طالب
        - قسّم لـ 10 ملفات (50 ورقة لكل ملف)
        - كل ملف: 5 دفعات × 10 أوراق = 3-4 دقائق
        - الإجمالي: 30-40 دقيقة ✅
        
        **الوقت المتوقع:** 10 ملفات × 3-4 دقائق = 30-40 دقيقة
        **التكلفة:** 500 × $0.003 = $1.50
        """)
        
        sheets = st.file_uploader(
            "ارفع ملف PDF (⚠️ **أقصى حد: 50 صفحة**)",
            type=["pdf"],
            accept_multiple_files=False,
            key="sheets"
        )
        
        st.warning("⚠️ **حد الذاكرة:** لا ترفع ملفات أكبر من 50 صفحة! قسّم الملفات الكبيرة أولاً.")
        
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.slider("📦 Batch size", 5, 20, 10, help="للذاكرة المحدودة: استخدم 10 أو أقل")
        with col2:
            auto_continue = st.checkbox("🔄 Auto-continue", value=False, help="⚠️ أطفئه لو في مشاكل ذاكرة")
        
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
                        
                        # Convert and compress immediately
                        bgr = pil_to_bgr(page)
                        
                        # Extract code with AI (simple and reliable)
                        img = bgr_to_bytes(bgr)
                        res = analyze_with_ai(img, api_key, False)
                        
                        if not res.success or not res.student_code:
                            st.warning(f"⚠️ Page {i+1}: Failed to read")
                            del page, bgr, img
                            continue
                        
                        code = res.student_code.strip()
                        
                        # CRITICAL: Double-check if code seems wrong
                        needs_recheck = False
                        recheck_reason = ""
                        
                        if len(code) == 4 and code.isdigit():
                            code_int = int(code)
                            
                            # Suspicious patterns that need double-check
                            if code[0] == '0':
                                needs_recheck = True
                                recheck_reason = "starts with 0"
                            elif code_int > 1057:
                                needs_recheck = True
                                recheck_reason = "out of range"
                            elif not find_student_by_code(st.session_state.students, code):
                                needs_recheck = True
                                recheck_reason = "not in student list"
                        else:
                            needs_recheck = True
                            recheck_reason = "invalid format"
                        
                        # DOUBLE-CHECK: Re-read with ultra-detailed prompt
                        if needs_recheck:
                            st.warning(f"🔍 Page {i+1}: Code {code} suspicious ({recheck_reason}) - double-checking...")
                            
                            # Ultra-detailed prompt focusing ONLY on code
                            detailed_prompt = """⚠️ CRITICAL RE-CHECK: Student Code Grid Reading

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 FOCUS: CODE GRID ONLY (top-left area of page)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Look VERY CAREFULLY at the bubble grid. It has 4 VERTICAL COLUMNS:

COLUMN 1 (1st digit):  ⓪ ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨  → Which bubble is FILLED?
COLUMN 2 (2nd digit):  ⓪ ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨  → Which bubble is FILLED?
COLUMN 3 (3rd digit):  ⓪ ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨  → Which bubble is FILLED?
COLUMN 4 (4th digit):  ⓪ ① ② ③ ④ ⑤ ⑥ ⑦ ⑧ ⑨  → Which bubble is FILLED?

⚠️ CRITICAL WARNINGS:
• First column MUST be "1" (not 0, not 7)
• Valid codes: 1000-1057 ONLY
• Watch for similar-looking numbers:
  - 0 vs 8 (zero vs eight)
  - 1 vs 7 (one vs seven)
  - 3 vs 8 (three vs eight)
  - 5 vs 6 (five vs six)
  - 7 vs 9 (seven vs nine)

VERIFICATION STEPS:
1. Find the grid (top-left area)
2. Read Column 1 carefully → Usually "1"
3. Read Column 2 carefully → 0-9
4. Read Column 3 carefully → 0-9
5. Read Column 4 carefully → 0-9
6. Double-check: Does it look reasonable?
7. Verify: Is it between 1000-1057?

JSON ONLY:
{
  "col1": "1",
  "col2": "0",
  "col3": "1",
  "col4": "7",
  "student_code": "1017",
  "confidence": "high"
}"""
                            
                            # Second attempt with ultra-detailed prompt
                            import anthropic
                            client = anthropic.Anthropic(api_key=api_key)
                            image_b64 = base64.b64encode(img).decode('utf-8')
                            
                            message = client.messages.create(
                                model="claude-sonnet-4-20250514",
                                max_tokens=500,
                                messages=[{
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                                        {"type": "text", "text": detailed_prompt}
                                    ]
                                }]
                            )
                            
                            # Parse second attempt
                            try:
                                import json
                                text = message.content[0].text
                                # Remove markdown if present
                                if '```' in text:
                                    text = text.split('```')[1]
                                    if text.startswith('json'):
                                        text = text[4:]
                                text = text.strip()
                                
                                data = json.loads(text)
                                new_code = data.get('student_code', '').strip()
                                
                                if new_code and new_code != code:
                                    st.info(f"🔄 Page {i+1}: Double-check changed code: {code} → {new_code}")
                                    code = new_code
                                    res.student_code = new_code
                                else:
                                    st.warning(f"⚠️ Page {i+1}: Double-check confirmed: {code} (may need manual review)")
                            except:
                                st.error(f"❌ Page {i+1}: Double-check failed - keeping original: {code}")
                        
                        # Free memory immediately after AI processing
                        del page, bgr, img
                        
                        # Force garbage collection every 10 pages
                        if (i - current) % 10 == 0:
                            gc.collect()
                        
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
                    
                    st.session_state.current_file_idx = end
                    
                    # Aggressive memory cleanup
                    if end >= total:
                        # File complete - clear everything
                        del st.session_state.current_file_pages
                        del st.session_state.current_file_idx
                        gc.collect()
                    
                    gc.collect()
                    
                    st.success(f"✅ Processed {processed_count} pages")
                    
                    if end >= total:
                        st.balloons()
                        st.success("🎉 File complete!")
                    elif auto_continue:
                        time.sleep(0.5)
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
