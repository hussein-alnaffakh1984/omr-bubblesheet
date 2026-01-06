# OMR Bubble Sheet (Streamlit)

تصحيح ببل شيت أونلاين باستخدام Streamlit.

## المدخلات
- ملف Excel للطلاب: `student_code, student_name`
- Answer Key (PDF أو صورة)
- أوراق الطلاب (PDF متعدد الصفحات)

## المخرجات
ملف Excel يحتوي فقط:
- sheet_index
- student_code
- student_name
- score

## التشغيل محليًا
```bash
pip install -r requirements.txt
streamlit run app.py
