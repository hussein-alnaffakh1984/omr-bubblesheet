# app.py
import io
import math
import numpy as np
import pandas as pd
import cv2
import streamlit as st
from sklearn.cluster import DBSCAN

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


# =========================
# 0) Load PDF / Images
# =========================
def load_pages(file_bytes: bytes, filename: str, zoom: float = 4.0):
    """
    Returns list of BGR images for each page.
    zoom=4.0 IMPORTANT for thin outlines in PDF.
    """
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        if fitz is None:
            raise RuntimeError("PyMuPDF not installed. Add 'pymupdf' to requirements.txt")
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        mat = fitz.Matrix(zoom, zoom)
        for i in range(doc.page_count):
            pg = doc.load_page(i)
            pix = pg.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            pages.append(img)
        doc.close()
        return pages
    else:
        data = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Unsupported image format.")
        return [img]


# =========================
# 1) Preprocess & Warp Page
# =========================
def preprocess_page(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray2 = clahe.apply(gray)

    thr = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr, gray2

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    page = None
    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            page = approx
            break

    if page is None:
        return bgr, gray2

    pts = page.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)

    W = int(max(wA, wB))
    H = int(max(hA, hB))
    W = max(W, 1000)
    H = max(H, 1200)

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    warped = cv2.warpPerspective(bgr, M, (W, H))

    wgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    wgray = cv2.GaussianBlur(wgray, (5, 5), 0)
    wgray = clahe.apply(wgray)
    return warped, wgray


# =========================
# 2) Bubble Detection (Dual + Radius-band filter)
# =========================
def detect_bubbles(gray: np.ndarray):
    H, W = gray.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ---- A) contours for filled marks
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubblesA = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 25 or area > (H * W * 0.02):
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circularity = 4 * math.pi * area / (peri * peri + 1e-9)
        if circularity < 0.45:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h + 1e-9)
        if ar < 0.6 or ar > 1.4:
            continue
        cx = x + w / 2.0
        cy = y + h / 2.0
        r = (w + h) / 4.0
        bubblesA.append((cx, cy, r))

    # ---- B) Hough for thin outlines (reduce false circles)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=140,
        param2=26,     # أعلى = دوائر أقل (أكثر دقة)
        minRadius=8,
        maxRadius=55,
    )

    bubblesB = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if 0 <= x < W and 0 <= y < H and 8 <= r <= 70:
                bubblesB.append((float(x), float(y), float(r)))

    bubbles = bubblesA + bubblesB
    if len(bubbles) == 0:
        return np.zeros((0, 3), dtype=np.float32), thr, {"mode": "none", "r_med": 0.0}

    bubbles = np.array(bubbles, dtype=np.float32)

    # ✅ Radius-band filtering (يقتل 90% من false positives)
    r = bubbles[:, 2]
    r_med = float(np.median(r))
    lo = 0.70 * r_med
    hi = 1.35 * r_med
    keep = (r >= lo) & (r <= hi)
    bubbles = bubbles[keep]

    dbg = cv2.Canny(gray, 60, 160)
    dbg = cv2.dilate(dbg, kernel, iterations=1)

    meta = {"mode": "dual+radius_band", "r_med": r_med, "keep_lo": lo, "keep_hi": hi}
    return bubbles, dbg, meta


# =========================
# 3) Clustering (eps based on r_med)
# =========================
def cluster_points(bubbles: np.ndarray, r_med: float):
    if len(bubbles) < 30:
        return []
    eps = max(18.0, min(55.0, r_med * 2.4))
    min_samples = 10
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(bubbles[:, :2])
    labels = db.labels_

    clusters = []
    for lb in sorted(set(labels)):
        if lb == -1:
            continue
        idx = np.where(labels == lb)[0]
        if len(idx) >= 25:
            clusters.append(idx)
    return clusters


def count_levels(vals, tol):
    vals = np.sort(vals)
    levels = []
    for v in vals:
        if not levels or abs(v - levels[-1]) > tol:
            levels.append(v)
    return len(levels)


def estimate_rows_cols(xy: np.ndarray, r_med: float):
    tol = max(16.0, r_med * 1.8)
    cols = count_levels(xy[:, 0], tol)
    rows = count_levels(xy[:, 1], tol)
    return rows, cols


# =========================
# 4) Identify code grid (10x4) and answer grids
# =========================
def find_code_cluster(clusters_idx, bubbles, r_med: float):
    best = None
    best_score = 1e9
    for i, idx in enumerate(clusters_idx):
        xy = bubbles[idx][:, :2]
        rows, cols = estimate_rows_cols(xy, r_med)
        score = abs(rows - 10) * 3 + abs(cols - 4) * 6
        if score < best_score:
            best_score = score
            best = i

    if best is None:
        return None

    xy = bubbles[clusters_idx[best]][:, :2]
    rows, cols = estimate_rows_cols(xy, r_med)
    if abs(rows - 10) <= 2 and abs(cols - 4) <= 2:
        return best
    return None


def find_answer_clusters(clusters_idx, bubbles, r_med: float, code_cluster_i=None):
    cands = []
    for i, idx in enumerate(clusters_idx):
        if code_cluster_i is not None and i == code_cluster_i:
            continue
        xy = bubbles[idx][:, :2]
        rows, cols = estimate_rows_cols(xy, r_med)
        if rows >= 10 and cols in (4, 5, 6):
            cands.append(i)
    return cands


# =========================
# 5) Fill score
# =========================
def bubble_fill_score(gray: np.ndarray, cx: float, cy: float, r: float):
    H, W = gray.shape[:2]
    pad = int(max(10, r * 1.4))
    x0 = max(int(cx) - pad, 0)
    x1 = min(int(cx) + pad, W - 1)
    y0 = max(int(cy) - pad, 0)
    y1 = min(int(cy) + pad, H - 1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    thr = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )

    mh, mw = thr.shape[:2]
    mask = np.zeros((mh, mw), dtype=np.uint8)
    ccx = int(cx) - x0
    ccy = int(cy) - y0
    rr = int(max(4, r * 0.62))
    cv2.circle(mask, (ccx, ccy), rr, 255, -1)

    inside = thr[mask == 255]
    if inside.size == 0:
        return 0.0
    return float(np.mean(inside) / 255.0)


# =========================
# 6) Read student code (4 digits)
# =========================
def read_student_code(gray, code_bubbles, code_cols=4):
    pts = code_bubbles.copy()
    pts = pts[np.argsort(pts[:, 0])]  # by x

    col_bins = []
    col_center = []
    for i in range(len(pts)):
        x = pts[i, 0]
        if not col_center or abs(x - col_center[-1]) > 25:
            col_center.append(x)
            col_bins.append([i])
        else:
            col_bins[-1].append(i)
            col_center[-1] = float(np.mean(pts[col_bins[-1], 0]))

    col_bins = sorted(col_bins, key=len, reverse=True)[:code_cols]
    col_bins = sorted(col_bins, key=lambda b: float(np.mean(pts[b, 0])))

    digits = []
    for b in col_bins:
        col = pts[b]
        col = col[np.argsort(col[:, 1])]  # y: 0..9
        scores = [bubble_fill_score(gray, cx, cy, r) for cx, cy, r in col]
        k = int(np.argmax(scores))
        digits.append(str(k))
    return "".join(digits)


# =========================
# 7) Answers reading helpers
# =========================
def split_blocks_by_x(points_xy, gap_factor=2.6):
    xs = np.sort(points_xy[:, 0])
    if len(xs) < 30:
        return [np.arange(len(points_xy))]
    diffs = np.diff(xs)
    med = float(np.median(diffs)) if np.median(diffs) > 0 else 1.0
    cut = med * gap_factor
    cuts = np.where(diffs > cut)[0]
    if len(cuts) == 0:
        return [np.arange(len(points_xy))]
    sort_idx = np.argsort(points_xy[:, 0])
    boundaries = [0] + (cuts + 1).tolist() + [len(xs)]
    blocks = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        blocks.append(sort_idx[a:b])
    return blocks


def group_rows(points, y_tol):
    points = points[np.argsort(points[:, 1])]
    rows = []
    cur = [points[0]]
    for p in points[1:]:
        if abs(p[1] - cur[-1][1]) <= y_tol:
            cur.append(p)
        else:
            rows.append(np.array(cur))
            cur = [p]
    rows.append(np.array(cur))
    return rows


def pick_best_4_in_row(row):
    """
    If row has too many points due to noise, pick 4 representative points.
    """
    row = row[np.argsort(row[:, 0])]
    if len(row) <= 6:
        return row
    # pick 4 distributed indices
    idxs = [0, len(row)//3, (2*len(row))//3, len(row)-1]
    return row[idxs]


def read_answers(gray, ans_bubbles, r_med: float, option_letters="ABCDE"):
    pts = ans_bubbles.copy()
    blocks = split_blocks_by_x(pts[:, :2], gap_factor=2.7)
    blocks = sorted(blocks, key=lambda idxs: float(np.mean(pts[idxs, 0])))

    all_ans, all_conf = [], []
    y_tol = max(16.0, r_med * 1.9)

    for bidx in blocks:
        blk = pts[bidx]
        rows = group_rows(blk, y_tol=y_tol)

        for row in rows:
            row = pick_best_4_in_row(row)
            row = row[np.argsort(row[:, 0])]
            if len(row) < 3:
                continue

            scores = [bubble_fill_score(gray, cx, cy, r) for cx, cy, r in row]
            scores_sorted = sorted(scores, reverse=True)
            best = int(np.argmax(scores))
            second = scores_sorted[1] if len(scores_sorted) > 1 else 0.0
            conf = float(scores_sorted[0] - second)

            if scores_sorted[0] < 0.22:
                all_ans.append("BLANK")
                all_conf.append(0.0)
                continue
            if conf < 0.06:
                all_ans.append("MULTI")
                all_conf.append(conf)
                continue

            all_ans.append(option_letters[best] if best < len(option_letters) else str(best))
            all_conf.append(conf)

    return all_ans, all_conf


# =========================
# 8) Parse one page
# =========================
def parse_page(bgr):
    warped, gray = preprocess_page(bgr)
    bubbles, dbg_img, meta = detect_bubbles(gray)

    if bubbles is None or len(bubbles) < 60:
        return None, {
            "error": "bubbles_too_few",
            "bubbles": int(0 if bubbles is None else len(bubbles)),
            "meta": meta
        }

    r_med = float(np.median(bubbles[:, 2])) if len(bubbles) else 10.0
    clusters_idx = cluster_points(bubbles, r_med)

    if not clusters_idx:
        return None, {"error": "no_clusters", "bubbles": int(len(bubbles)), "r_med": r_med, "meta": meta}

    code_i = find_code_cluster(clusters_idx, bubbles, r_med)
    student_code = "UNKNOWN"
    if code_i is not None:
        code_cluster = bubbles[clusters_idx[code_i]]
        student_code = read_student_code(gray, code_cluster, code_cols=4)

    ans_ids = find_answer_clusters(clusters_idx, bubbles, r_med, code_cluster_i=code_i)
    if not ans_ids:
        return None, {"error": "no_answer_cluster", "bubbles": int(len(bubbles)), "r_med": r_med, "clusters": int(len(clusters_idx)), "meta": meta}

    parsed = []
    for i in ans_ids:
        ans_cluster = bubbles[clusters_idx[i]]
        ans, conf = read_answers(gray, ans_cluster, r_med=r_med, option_letters="ABCDE")
        parsed.append((i, ans, conf, len(ans_cluster)))

    parsed.sort(key=lambda t: len(t[1]), reverse=True)

    best_ans, best_conf = parsed[0][1], parsed[0][2]
    merged_ans = best_ans[:]
    merged_conf = best_conf[:]

    for (_, ans, conf, _) in parsed[1:]:
        if len(ans) >= 10:
            merged_ans += ans
            merged_conf += conf

    return {
        "student_code": student_code,
        "answers": merged_ans,
        "confidence": merged_conf,
        "num_answers": len(merged_ans)
    }, {
        "bubbles": int(len(bubbles)),
        "r_med": r_med,
        "clusters": int(len(clusters_idx)),
        "mode": meta.get("mode"),
        "radius_band": [meta.get("keep_lo"), meta.get("keep_hi")]
    }


# =========================
# 9) Scoring
# =========================
def score_answers(student_answers, key_answers):
    n = min(len(student_answers), len(key_answers))
    correct = 0
    for i in range(n):
        if student_answers[i] == key_answers[i]:
            correct += 1
    return correct, n


# =========================
# 10) Streamlit UI
# =========================
st.set_page_config(page_title="Smart OMR", layout="wide")
st.title("📄 Smart OMR — قراءة الإجابات + كود الطالب (4 خانات ثابت)")

st.info(
    "الخطوات:\n"
    "1) ارفع Answer Key (ورقة مظللة)\n"
    "2) ارفع أوراق الطلاب (متعدد)\n\n"
    "ملاحظة: إذا رفعت قالب فارغ كـ Answer Key فسيتم اكتشاف الشبكة، لكن الإجابات ستكون BLANK."
)

c1, c2 = st.columns(2)
with c1:
    key_file = st.file_uploader("1) Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
with c2:
    student_files = st.file_uploader("2) أوراق الطلاب (PDF/صور) - متعدد", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

key_answers = None

if key_file is not None:
    try:
        pages = load_pages(key_file.getvalue(), key_file.name, zoom=4.0)
        res, dbg = parse_page(pages[0])

        if res is None:
            st.error(f"فشل قراءة Answer Key: {dbg}")
        else:
            blanks = sum(1 for a in res["answers"] if a == "BLANK")
            total = res["num_answers"]

            if total == 0:
                st.error("تم العثور على فقاعات لكن لم يتم بناء شبكة أسئلة صحيحة. راجع الـ debug بالأسفل.")
                st.write(dbg)
            elif blanks / max(1, total) > 0.85:
                st.warning(
                    f"تم اكتشاف شبكة أسئلة (عدد={total}) لكن أغلبها BLANK.\n"
                    "هذا غالبًا قالب فارغ وليس Answer Key مظلّل."
                )
                st.write({"detected_questions": total, "student_code_detected": res["student_code"], "debug": dbg})
            else:
                key_answers = res["answers"]
                st.success(f"✅ تم استخراج Answer Key بنجاح | الأسئلة: {len(key_answers)}")
                st.write({"detected_questions": len(key_answers), "student_code_in_key": res["student_code"], "debug": dbg})

    except Exception as e:
        st.exception(e)

if key_answers and student_files:
    results = []
    for f in student_files:
        try:
            pages = load_pages(f.getvalue(), f.name, zoom=4.0)
            for pi, page in enumerate(pages):
                res, dbg = parse_page(page)
                if res is None:
                    results.append({
                        "file": f.name,
                        "page": pi + 1,
                        "student_code": "UNKNOWN",
                        "score": 0,
                        "total": len(key_answers),
                        "detected_answers": 0,
                        "mean_confidence": 0.0,
                        "notes": str(dbg)
                    })
                    continue

                correct, total = score_answers(res["answers"], key_answers)
                mean_conf = float(np.mean(res["confidence"])) if res["confidence"] else 0.0

                results.append({
                    "file": f.name,
                    "page": pi + 1,
                    "student_code": res["student_code"],
                    "score": correct,
                    "total": total,
                    "detected_answers": res["num_answers"],
                    "mean_confidence": mean_conf,
                    "notes": f"bubbles={dbg['bubbles']} r_med={dbg['r_med']:.2f} clusters={dbg['clusters']} mode={dbg['mode']}"
                })

        except Exception as e:
            results.append({
                "file": f.name,
                "page": 1,
                "student_code": "UNKNOWN",
                "score": 0,
                "total": len(key_answers),
                "detected_answers": 0,
                "mean_confidence": 0.0,
                "notes": f"ERROR: {e}"
            })

    df = pd.DataFrame(results)
    st.subheader("📊 النتائج")
    st.dataframe(df, use_container_width=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")

    st.download_button(
        "⬇️ تنزيل Excel النتائج",
        data=output.getvalue(),
        file_name="omr_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
