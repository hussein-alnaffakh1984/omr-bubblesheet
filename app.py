# app.py
import io, os, math
import streamlit as st
import numpy as np
import pandas as pd
import cv2

from sklearn.cluster import DBSCAN

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# -----------------------------
# Utils: PDF/Image loading
# -----------------------------
def load_pages(file_bytes: bytes, filename: str, zoom: float = 2.0):
    """
    Returns list of BGR images (np.ndarray) for each page.
    Supports images and PDFs.
    """
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) not installed. Add 'pymupdf' to requirements.")
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        mat = fitz.Matrix(zoom, zoom)
        for i in range(doc.page_count):
            pg = doc.load_page(i)
            pix = pg.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            # PyMuPDF gives RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            pages.append(img)
        doc.close()
        return pages
    else:
        data = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Unsupported image format or corrupted file.")
        return [img]

# -----------------------------
# Preprocess / Deskew / Warp
# -----------------------------
def preprocess_page(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # binary for contour finding
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )

    # find page contour (largest rectangle-ish)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr, gray

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    page = None
    for c in cnts[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            page = approx
            break

    if page is None:
        # if failed, return original
        return bgr, gray

    # order points
    pts = page.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)

    W = int(max(wA, wB))
    H = int(max(hA, hB))
    W = max(W, 800)
    H = max(H, 1000)

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    warped = cv2.warpPerspective(bgr, M, (W, H))
    wgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    wgray = cv2.GaussianBlur(wgray, (5, 5), 0)
    return warped, wgray

# -----------------------------
# Bubble detection
# -----------------------------
def detect_bubble_candidates(gray: np.ndarray):
    # binary for circles
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )

    # remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = gray.shape[:2]
    bubbles = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 80 or area > (H * W * 0.01):
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circularity = 4 * math.pi * area / (peri * peri + 1e-9)
        if circularity < 0.55:
            continue

        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h + 1e-9)
        if ar < 0.7 or ar > 1.3:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        r = (w + h) / 4.0
        bubbles.append((cx, cy, r))
    return np.array(bubbles, dtype=np.float32), thr

# -----------------------------
# Grid clustering helpers
# -----------------------------
def cluster_points_xy(bubbles_xy: np.ndarray, eps: float = 28.0, min_samples: int = 10):
    """
    bubbles_xy: (N,2)
    returns list of clusters, each cluster is indices
    """
    if len(bubbles_xy) == 0:
        return []
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(bubbles_xy)
    labels = db.labels_
    clusters = []
    for lb in sorted(set(labels)):
        if lb == -1:
            continue
        idx = np.where(labels == lb)[0]
        clusters.append(idx)
    return clusters

def estimate_grid_dims(cluster_xy: np.ndarray, y_tol: float = 18.0, x_tol: float = 18.0):
    """
    Estimate rows/cols by unique-ish y and x levels.
    """
    xs = np.sort(cluster_xy[:, 0])
    ys = np.sort(cluster_xy[:, 1])

    def count_levels(vals, tol):
        levels = []
        for v in vals:
            if not levels or abs(v - levels[-1]) > tol:
                levels.append(v)
        return len(levels)

    cols = count_levels(xs, x_tol)
    rows = count_levels(ys, y_tol)
    return rows, cols

def split_into_blocks_by_x(cluster_xy: np.ndarray, gap_factor: float = 2.0):
    """
    If answers are in multiple blocks (columns), split by big gaps on x.
    """
    xs = np.sort(cluster_xy[:, 0])
    if len(xs) < 10:
        return [np.arange(len(cluster_xy))]

    diffs = np.diff(xs)
    med = np.median(diffs) if np.median(diffs) > 0 else 1.0
    cut = med * gap_factor
    cut_positions = np.where(diffs > cut)[0]

    if len(cut_positions) == 0:
        return [np.arange(len(cluster_xy))]

    # build ranges
    boundaries = [0] + (cut_positions + 1).tolist() + [len(xs)]
    # map sorted indices to original
    sort_idx = np.argsort(cluster_xy[:, 0])
    blocks = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        blk_sorted = sort_idx[a:b]
        blocks.append(blk_sorted)
    return blocks

# -----------------------------
# Fill score
# -----------------------------
def bubble_fill_score(gray: np.ndarray, cx: float, cy: float, r: float):
    """
    Score = black pixel ratio inside inner circle after adaptive threshold.
    """
    H, W = gray.shape[:2]
    x0 = int(max(cx - r * 1.2, 0)); x1 = int(min(cx + r * 1.2, W - 1))
    y0 = int(max(cy - r * 1.2, 0)); y1 = int(min(cy + r * 1.2, H - 1))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    thr = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )

    # circle mask
    mh, mw = thr.shape[:2]
    mask = np.zeros((mh, mw), dtype=np.uint8)
    ccx = int(cx - x0); ccy = int(cy - y0)
    rr = int(max(3, r * 0.55))
    cv2.circle(mask, (ccx, ccy), rr, 255, -1)

    inside = thr[mask == 255]
    if inside.size == 0:
        return 0.0
    return float(np.mean(inside) / 255.0)  # 0..1 (higher = more filled)

# -----------------------------
# Identify code grid (10x4) and answer grids (4/5 cols)
# -----------------------------
def find_code_cluster(clusters, bubbles, tol_rows=10, tol_cols=4):
    """
    Return best cluster index for code grid: ~10 rows and ~4 cols
    """
    best = None
    best_dist = 1e9
    for i, idx in enumerate(clusters):
        xy = bubbles[idx][:, :2]
        rows, cols = estimate_grid_dims(xy)
        dist = abs(rows - tol_rows) + abs(cols - tol_cols) * 2
        if dist < best_dist:
            best_dist = dist
            best = i
    # accept only if close enough
    if best is None:
        return None
    xy = bubbles[clusters[best]][:, :2]
    rows, cols = estimate_grid_dims(xy)
    if abs(rows - tol_rows) <= 2 and abs(cols - tol_cols) <= 2:
        return best
    return None

def find_answer_clusters(clusters, bubbles):
    """
    Answer clusters are those with many rows and 4-5 cols.
    """
    answer = []
    for i, idx in enumerate(clusters):
        xy = bubbles[idx][:, :2]
        rows, cols = estimate_grid_dims(xy)
        if rows >= 10 and cols in (4, 5, 6):  # allow slight miscount
            answer.append(i)
    return answer

# -----------------------------
# Read student code (4 digits)
# -----------------------------
def read_student_code(gray, bubbles_cluster, code_cols=4):
    pts = bubbles_cluster.copy()
    # sort by x then group columns
    order = np.argsort(pts[:, 0])
    pts = pts[order]

    # cluster columns by x proximity
    x_vals = pts[:, 0]
    col_centers = []
    col_bins = []
    for i in range(len(x_vals)):
        if not col_centers or abs(x_vals[i] - col_centers[-1]) > 22:
            col_centers.append(x_vals[i])
            col_bins.append([i])
        else:
            col_bins[-1].append(i)
            col_centers[-1] = np.mean(x_vals[col_bins[-1]])

    # keep most relevant columns (pick 4 most populated)
    col_bins = sorted(col_bins, key=len, reverse=True)[:code_cols]
    # sort columns left->right
    col_bins = sorted(col_bins, key=lambda b: np.mean(x_vals[b]))

    digits = []
    for b in col_bins:
        col_pts = pts[b]
        # sort by y (0..9)
        col_pts = col_pts[np.argsort(col_pts[:, 1])]
        # pick bubble with max fill
        scores = [bubble_fill_score(gray, cx, cy, r) for cx, cy, r in col_pts]
        k = int(np.argmax(scores))
        # digit = row index (0..9)
        digits.append(str(k))
    return "".join(digits)

# -----------------------------
# Read answers from multiple blocks
# -----------------------------
def read_answers_from_cluster(gray, bubbles_cluster, option_letters="ABCDE"):
    """
    bubbles_cluster: Nx3 (cx,cy,r)
    returns answers list (for this cluster), confidences list
    """
    pts = bubbles_cluster.copy()
    # group into blocks by X gaps (for multi-columns)
    blocks_idx = split_into_blocks_by_x(pts[:, :2], gap_factor=2.4)

    all_answers = []
    all_conf = []

    # process each block left->right
    for bidx in sorted(blocks_idx, key=lambda idxs: np.mean(pts[idxs, 0])):
        blk = pts[bidx]
        # group rows by y
        blk = blk[np.argsort(blk[:, 1])]

        rows = []
        current = [blk[0]]
        for p in blk[1:]:
            if abs(p[1] - current[-1][1]) <= 18:
                current.append(p)
            else:
                rows.append(np.array(current))
                current = [p]
        rows.append(np.array(current))

        # each row should have 4/5 bubbles: sort by x
        for row in rows:
            row = row[np.argsort(row[:, 0])]
            if len(row) < 3:
                continue

            scores = [bubble_fill_score(gray, cx, cy, r) for cx, cy, r in row]
            best = int(np.argmax(scores))
            # confidence = best - second best
            s_sorted = sorted(scores, reverse=True)
            conf = float(s_sorted[0] - (s_sorted[1] if len(s_sorted) > 1 else 0.0))

            # detect MULTI / BLANK
            if s_sorted[0] < 0.25:
                all_answers.append("BLANK")
                all_conf.append(0.0)
                continue
            if conf < 0.06:
                all_answers.append("MULTI")
                all_conf.append(conf)
                continue

            all_answers.append(option_letters[best] if best < len(option_letters) else str(best))
            all_conf.append(conf)

    return all_answers, all_conf

# -----------------------------
# High level: parse one page
# -----------------------------
def parse_omr_page(bgr):
    warped, gray = preprocess_page(bgr)
    bubbles, _ = detect_bubble_candidates(gray)

    if len(bubbles) < 50:
        return None, {"error": "bubbles_too_few", "bubbles": int(len(bubbles))}

    clusters_idx = cluster_points_xy(bubbles[:, :2], eps=28, min_samples=10)
    clusters = [bubbles[idx] for idx in clusters_idx]

    code_i = find_code_cluster(clusters_idx, bubbles, tol_rows=10, tol_cols=4)

    student_code = "UNKNOWN"
    if code_i is not None:
        code_cluster = clusters[code_i]
        student_code = read_student_code(gray, code_cluster, code_cols=4)

    answer_ids = find_answer_clusters(clusters_idx, bubbles)
    # remove code cluster if it was included
    if code_i is not None and code_i in answer_ids:
        answer_ids = [i for i in answer_ids if i != code_i]

    if not answer_ids:
        return None, {"error": "no_answer_cluster"}

    # choose the biggest answer cluster(s) by size, but sometimes answers split.
    # We'll read from all candidate answer clusters then keep the one with most answers,
    # OR merge if more than one gives a coherent total.
    parsed = []
    for i in answer_ids:
        ans, conf = read_answers_from_cluster(gray, clusters[i], option_letters="ABCDE")
        parsed.append((i, ans, conf))

    # pick best by max answers extracted
    parsed.sort(key=lambda t: len(t[1]), reverse=True)

    # Sometimes there are 2 separate answer zones (e.g., theoretical + practical).
    # We'll try to merge if second has >=10 answers.
    best_ans, best_conf = parsed[0][1], parsed[0][2]
    merged_ans, merged_conf = best_ans[:], best_conf[:]
    for _, ans, conf in parsed[1:]:
        if len(ans) >= 10:
            merged_ans += ans
            merged_conf += conf

    return {
        "student_code": student_code,
        "answers": merged_ans,
        "confidence": merged_conf,
        "num_answers": len(merged_ans),
    }, {"bubbles": int(len(bubbles)), "clusters": int(len(clusters_idx))}

# -----------------------------
# Scoring
# -----------------------------
def score_answers(student_answers, key_answers):
    n = min(len(student_answers), len(key_answers))
    correct = 0
    for i in range(n):
        if student_answers[i] == key_answers[i]:
            correct += 1
    return correct, n

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Smart OMR (Template-Free)", layout="wide")
st.title("📄 Smart OMR - يقرأ الإجابات + كود الطالب (4 خانات ثابت)")

st.info("ارفع مفتاح الإجابة أولاً (Answer Key)، بعدها ارفع أوراق الطلاب. البرنامج يستخرج كود الطالب ويصحح ويصدّر Excel.")

col1, col2 = st.columns(2)

with col1:
    key_file = st.file_uploader("1) ارفع Answer Key (PDF أو صورة)", type=["pdf", "png", "jpg", "jpeg"])
with col2:
    student_files = st.file_uploader("2) ارفع أوراق الطلاب (PDF/صور) - متعدد", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

key_answers = None

if key_file is not None:
    try:
        pages = load_pages(key_file.getvalue(), key_file.name)
        # parse first page as key
        res, dbg = parse_omr_page(pages[0])
        if res is None:
            st.error(f"فشل قراءة Answer Key: {dbg}")
        else:
            key_answers = res["answers"]
            st.success(f"✅ تم استخراج Answer Key بنجاح. عدد الأسئلة المكتشفة: {len(key_answers)}")
            st.write({"detected_questions": len(key_answers), "student_code_in_key": res["student_code"], "debug": dbg})
    except Exception as e:
        st.exception(e)

if key_answers and student_files:
    rows = []
    for f in student_files:
        try:
            pages = load_pages(f.getvalue(), f.name)
            for pi, page in enumerate(pages):
                res, dbg = parse_omr_page(page)
                if res is None:
                    rows.append({
                        "file": f.name,
                        "page": pi+1,
                        "student_code": "UNKNOWN",
                        "score": 0,
                        "total": len(key_answers),
                        "notes": str(dbg),
                    })
                    continue

                correct, total = score_answers(res["answers"], key_answers)
                rows.append({
                    "file": f.name,
                    "page": pi+1,
                    "student_code": res["student_code"],
                    "score": correct,
                    "total": total,
                    "detected_answers": res["num_answers"],
                    "mean_confidence": float(np.mean(res["confidence"])) if res["confidence"] else 0.0,
                    "notes": f"bubbles={dbg.get('bubbles')} clusters={dbg.get('clusters')}",
                })
        except Exception as e:
            rows.append({"file": f.name, "page": 1, "student_code": "UNKNOWN", "score": 0, "total": len(key_answers), "notes": f"ERROR: {e}"})

    df = pd.DataFrame(rows)
    st.subheader("📊 النتائج")
    st.dataframe(df, use_container_width=True)

    # Export Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    st.download_button(
        "⬇️ تنزيل Excel النتائج",
        data=output.getvalue(),
        file_name="omr_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
