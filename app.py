def read_code_auto(page_bgr: np.ndarray) -> CodeResult:
    H, W = page_bgr.shape[:2]
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ---------- helper: detect circles (contours first, then hough fallback) ----------
    def detect_circles(gray_img):
        edges = cv2.Canny(gray_img, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circles = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue
            peri = cv2.arcLength(c, True) + 1e-6
            circ = 4 * np.pi * area / (peri * peri)
            if circ < MIN_CIRCULARITY:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < R_MIN or r > R_MAX:
                continue
            circles.append((float(x), float(y), float(r)))

        # Fallback: HoughCircles if contour-based failed
        if len(circles) < 60:
            g = cv2.medianBlur(gray_img, 5)
            hc = cv2.HoughCircles(
                g, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=18,
                param1=120, param2=18,
                minRadius=R_MIN, maxRadius=R_MAX
            )
            if hc is not None:
                hc = np.squeeze(hc, axis=0)
                circles = [(float(x), float(y), float(r)) for x, y, r in hc]

        return circles

    circles = detect_circles(gray)
    if len(circles) < 50:
        return CodeResult(None, False, "REVIEW: not enough bubbles detected")

    circles = np.array(circles, dtype=np.float32)
    rs = circles[:, 2]
    r_med = float(np.median(rs))
    keep = (rs > r_med * 0.65) & (rs < r_med * 1.55)
    circles = circles[keep]
    if len(circles) < 45:
        return CodeResult(None, False, "REVIEW: bubble size filtering failed")

    # Focus on top half (where code exists)
    top = circles[circles[:, 1] < TOP_REGION_RATIO * H]
    pts = top if len(top) >= 40 else circles

    x = pts[:, 0]
    y = pts[:, 1]

    # Cluster rows and columns
    try:
        row_labels, row_centers = kmeans_1d(y, 4)
    except Exception:
        return CodeResult(None, False, "REVIEW: row clustering failed")

    # keep points close to their row center
    ydist = np.abs(y - row_centers[row_labels])
    thr = np.percentile(ydist, 70) * 2.0 + 1e-6
    good = ydist < thr

    pts2 = pts[good]
    row_labels2 = row_labels[good]
    if len(pts2) < 35:
        return CodeResult(None, False, "REVIEW: row grid unstable")

    try:
        col_labels, col_centers = kmeans_1d(pts2[:, 0], 10)
    except Exception:
        return CodeResult(None, False, "REVIEW: column clustering failed")

    # Build 4x10 grid
    grid = [[None for _ in range(10)] for _ in range(4)]
    for (cx, cy, r), rr, cc in zip(pts2, row_labels2, col_labels):
        dx = abs(cx - col_centers[cc])
        dy = abs(cy - row_centers[rr])
        d = dx + dy
        if grid[rr][cc] is None or d < grid[rr][cc][0]:
            grid[rr][cc] = (d, int(cx), int(cy), int(r))

    missing = sum(1 for rr in range(4) for cc in range(10) if grid[rr][cc] is None)
    if missing > 10:
        return CodeResult(None, False, "REVIEW: code grid not found")

    # ---------- Adaptive decision per row ----------
    digits = []
    for rr in range(4):
        scores = np.zeros((10,), dtype=np.float32)

        for cc in range(10):
            cell = grid[rr][cc]
            if cell is None:
                scores[cc] = 0.0
                continue
            _, cx, cy, r = cell
            scores[cc] = ink_score_in_circle(gray, cx, cy, r)

        best = int(np.argmax(scores))
        best_sc = float(scores[best])

        # robust stats
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)) + 1e-6)

        # adaptive thresholds:
        # 1) must stand out from background
        if best_sc < med + 3.0 * mad:
            return CodeResult(None, False, f"REVIEW: row {rr+1} faint/unclear")

        # 2) margin relative to row strength (not absolute)
        sorted_scores = np.sort(scores)
        second_sc = float(sorted_scores[-2])
        margin = best_sc - second_sc

        # relative margin: at least 12% of best, OR 2*MAD
        if margin < max(0.12 * best_sc, 2.0 * mad):
            return CodeResult(None, False, f"REVIEW: row {rr+1} ambiguous")

        digits.append(best)

    code = "".join(map(str, digits))
    if not code.isdigit():
        return CodeResult(None, False, "REVIEW: invalid code")

    code_int = int(code)
    if not (CODE_MIN <= code_int <= CODE_MAX):
        return CodeResult(None, False, f"REVIEW: code out of range ({code})")

    return CodeResult(code, True, "OK")
