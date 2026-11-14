"""COMP482 SudokuScan. Harris-derived corners; no solver-based OCR correction."""

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import cv2
import numpy as np
import pytesseract
from PIL import Image

SIZE = 450

# Boundary coordinates: bottom/right boundaries lie at 450, outside pixel 449.
DST = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])

OCR = "--oem 3 --psm 10 -c tessedit_char_whitelist=123456789"


def odd(x):
    n = max(3, round(x))
    return n if n % 2 else n + 1


def preprocess(image):
    scale = min(1.0, 1200 / max(image.shape[:2]))

    work = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else image.copy()
    )

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    size = odd(min(gray.shape) * 0.004)
    blur = cv2.GaussianBlur(gray, (size, size), 0)

    bg_size = odd(min(gray.shape) * 0.12)
    background = cv2.GaussianBlur(blur, (bg_size, bg_size), 0)
    normalized = cv2.divide(blur, background, scale=230)

    block = odd(min(gray.shape) * 0.035)
    ink = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        9,
    )

    return work, normalized, ink


def harris(gray, block=2, aperture=3, k=0.04, threshold=0.002):
    response = cv2.cornerHarris(np.float32(gray), block, aperture, k)
    peak = float(response.max())

    if peak <= 0:
        return response, np.empty((0, 2), np.float32)

    n = odd(min(gray.shape) * 0.009)
    maxima = cv2.dilate(response, np.ones((n, n), np.uint8))
    mask = (response >= peak * threshold) & (response >= maxima - 1e-5)

    # Collapse equal-response plateaus to one actual Harris peak each.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )

    peaks = []
    for label in range(1, count):
        x, y, w, h, _ = stats[label]
        yy, xx = np.where(labels[y:y + h, x:x + w] == label)
        index = int(np.argmax(response[y + yy, x + xx]))
        peaks.append((x + xx[index], y + yy[index]))

    points = np.asarray(peaks, dtype=np.float32).reshape(-1, 2)
    return response, points


def ordered(quad):
    q = np.asarray(quad, np.float32).reshape(4, 2)
    center = q.mean(axis=0)

    angles = np.arctan2(q[:, 1] - center[1], q[:, 0] - center[0])
    q = q[np.argsort(angles)]
    start = int(np.argmin(q.sum(axis=1)))
    q = np.roll(q, -start, axis=0)

    if cv2.contourArea(q.reshape(-1, 1, 2), oriented=True) < 0:
        q = q[[0, 3, 2, 1]]

    return q


def grid_score(ink, corners):
    matrix = cv2.getPerspectiveTransform(np.float32(corners), DST)
    warped = cv2.warpPerspective(ink, matrix, (SIZE, SIZE))
    binary = (warped > 0).astype(np.float32)

    scores = []
    for profile in (binary.mean(axis=0), binary.mean(axis=1)):
        lines = [
            float(profile[p - 5:p + 6].max())
            for p in range(50, 450, 50)
        ]
        gaps = [
            float(profile[p - 2:p + 3].mean())
            for p in range(25, 450, 50)
        ]
        scores.append(max(0.0, float(np.mean(lines) - np.mean(gaps))))

    return float(np.mean(scores))


def select_corners(ink, points):
    h, w = ink.shape

    closed = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    contours, _ = cv2.findContours(
        closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    options = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (0.015 * h * w < area < 0.9 * h * w):
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)

        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        q = ordered(approx)
        sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)

        if min(sides) / max(max(sides), 1) >= 0.28:
            options.append((area, q))

    options.sort(key=lambda item: -item[0])
    best = None

    for area, q in options[:50]:
        if len(points) < 4:
            break

        sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
        limit = 0.1 * float(sides.mean())
        indices = []

        for xy in q:
            distances = np.linalg.norm(points - xy, axis=1)
            index = int(distances.argmin())

            if distances[index] > limit:
                break

            indices.append(index)

        if len(indices) != 4 or len(set(indices)) != 4:
            continue

        # Final coordinates come exclusively from Harris candidates.
        corners = ordered(points[indices])

        if not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
            continue

        confidence = grid_score(ink, corners)
        if confidence < 0.07:
            continue

        rank = confidence + 0.02 * area / (h * w)
        if best is None or rank > best[0]:
            best = (rank, corners)

    return None if best is None else best[1]


def save_harris(work, response, points, corners, filename):
    heat = np.log1p(np.maximum(response, 0))
    nonzero = heat[heat > 0]
    upper = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0

    normalized = np.uint8(
        255 * np.clip(heat / max(upper, 1e-8), 0, 1)
    )
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    preview = cv2.addWeighted(work, 0.65, colored, 0.35, 0)

    for x, y in points:
        cv2.circle(preview, (round(x), round(y)), 2, (0, 0, 255), -1)

    if corners is not None:
        for label, (x, y) in zip(("TL", "TR", "BR", "BL"), corners):
            cv2.circle(
                preview, (round(x), round(y)), 7, (0, 255, 0), 2
            )
            cv2.putText(
                preview,
                label,
                (round(x) + 5, round(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    cv2.imwrite(str(filename), preview)


def rectify(image, corners):
    src = np.float32(corners)

    a3 = cv2.getAffineTransform(
        src[[0, 1, 3]], DST[[0, 1, 3]]
    )

    initial, _ = cv2.estimateAffine2D(src, DST, method=cv2.LMEDS)
    if initial is None:
        raise ValueError("Affine estimation failed")

    # Robust estimators may discard a point. Fit ALL four for A4 LS.
    design = np.column_stack([src.astype(float), np.ones(4)])
    a4 = np.linalg.lstsq(design, DST.astype(float), rcond=None)[0].T
    h4 = cv2.getPerspectiveTransform(src, DST)

    br = a3 @ np.r_[src[2], 1]
    e3 = float(np.linalg.norm(br - DST[2]))
    e4 = float(
        np.sqrt(np.mean(np.sum((design @ a4.T - DST) ** 2, axis=1)))
    )

    warps = {
        name: cv2.warpAffine(
            image, mat, (SIZE, SIZE), borderValue=(255,) * 3
        )
        for name, mat in [("A3", a3), ("A4", a4)]
    }
    warps["H4"] = cv2.warpPerspective(
        image, h4, (SIZE, SIZE), borderValue=(255,) * 3
    )

    return warps, a3, a4, h4, e3, e4


def cell_glyph(cell, prepared=True):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Avoid grid-line borders.
    roi = gray[8:42, 8:42]
    smooth = roi.copy()

    # Otsu alone can split blank paper texture into false foreground.
    background = float(np.percentile(smooth, 90))
    otsu, _ = cv2.threshold(
        smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    mask = np.uint8(
        smooth < min(otsu + 1, background - 28)
    ) * 255

    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        mask, 8
    )

    selected_candidates = []

    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        cx, cy = centers[i]

        if area < 12 or h < 8 or w < 2 or w > 29 or h > 31:
            continue

        if not (5 <= cx <= 29 and 5 <= cy <= 29):
            continue

        if (x <= 0 and w >= 25) or (y <= 0 and h >= 28):
            continue

        selected_candidates.append(i)

    if not selected_candidates:
        return None

    main = max(
        selected_candidates,
        key=lambda i: stats[i, cv2.CC_STAT_AREA],
    )
    x, y, w, h, area = map(int, stats[main])

    if area < 24 or h < 11:
        return None

    selected = [main]

    for i in selected_candidates:
        if i == main:
            continue

        xx, yy, ww, hh, _ = map(int, stats[i])
        dx = max(0, x - (xx + ww), xx - (x + w))
        dy = max(0, y - (yy + hh), yy - (y + h))

        if dx <= 2 and dy <= 2:
            selected.append(i)

    glyph_mask = np.uint8(np.isin(labels, selected)) * 255
    ys, xs = np.where(glyph_mask > 0)

    if len(xs) < 24:
        return None

    black = float(np.percentile(roi[glyph_mask > 0], 10))
    clean = np.uint8(
        np.clip(
            (roi.astype(float) - black) * 255
            / max(background - black, 1),
            0,
            255,
        )
    )
    clean[
        cv2.dilate(glyph_mask, np.ones((3, 3), np.uint8)) == 0
    ] = 255

    digit = 255 - clean[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1,
    ]

    if digit.shape[0] < 11:
        return None

    if not prepared:
        return cv2.resize(
            roi, (80, 80), interpolation=cv2.INTER_LINEAR
        )

    scale = 52 / max(digit.shape)
    new_w = max(1, round(digit.shape[1] * scale))
    new_h = max(1, round(digit.shape[0] * scale))

    resized = cv2.resize(
        digit, (new_w, new_h), interpolation=cv2.INTER_CUBIC
    )

    canvas = np.full((80, 80), 255, np.uint8)
    left = (80 - new_w) // 2
    top = (80 - new_h) // 2

    canvas[top:top + new_h, left:left + new_w] = 255 - resized
    return canvas


def glyph_variant(glyph, size):
    ys, xs = np.where(glyph < 128)

    if len(xs) == 0:
        return np.full((80, 80), 255, np.uint8)

    digit = glyph[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1,
    ]

    h, w = digit.shape
    scale = size / max(h, w)
    w = max(1, round(w * scale))
    h = max(1, round(h * scale))

    digit = cv2.resize(
        digit, (w, h), interpolation=cv2.INTER_CUBIC
    )

    canvas = np.full((80, 80), 255, np.uint8)
    y, x = (80 - h) // 2, (80 - w) // 2
    canvas[y:y + h, x:x + w] = digit
    return canvas


def ocr_pages(pages, psm=10):
    # One character per TIFF page; one engine startup for many cells.
    if not pages:
        return []

    images = [Image.fromarray(p) for p in pages]

    with tempfile.TemporaryDirectory(prefix="sudokuscan_") as directory:
        filename = str(Path(directory) / "cells.tif")

        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            compression="tiff_deflate",
            dpi=(300, 300),
        )

        data = pytesseract.image_to_data(
            filename,
            config=(
                f"--oem 3 --psm {psm} "
                "-c tessedit_char_whitelist=123456789"
            ),
            output_type=pytesseract.Output.DICT,
            timeout=5,
        )

    tokens = [[] for _ in pages]

    for i, token in enumerate(data["text"]):
        page = int(data["page_num"][i]) - 1
        if token.strip() and 0 <= page < len(tokens):
            tokens[page].append(
                (token.strip(), float(data["conf"][i]))
            )

    result = []

    for values in tokens:
        text = "".join(v[0] for v in values)
        result.append(
            (int(text), max(v[1] for v in values))
            if len(text) == 1 and text in "123456789"
            else (0, 0.0)
        )

    return result


def read_digit(glyph, alternative=False):
    return ocr_pages([glyph], 13 if alternative else 10)[0][0]


def candidate_cells(image, prepared=True):
    found = []

    for r in range(9):
        for c in range(9):
            cell = image[
                r * 50:(r + 1) * 50,
                c * 50:(c + 1) * 50,
            ]

            glyph = cell_glyph(cell, prepared)

            if glyph is not None:
                found.append((r, c, glyph))

    return found


def conflicts_in(grid):
    groups = []

    for i in range(9):
        groups.append([(i, c) for c in range(9)])
        groups.append([(r, i) for r in range(9)])

    for rr in (0, 3, 6):
        for cc in (0, 3, 6):
            groups.append(
                [
                    (rr + dr, cc + dc)
                    for dr in range(3)
                    for dc in range(3)
                ]
            )

    errors = set()

    for group in groups:
        seen = {}

        for r, c in group:
            value = grid[r][c]
            if value:
                seen.setdefault(value, []).append((r, c))

        for positions in seen.values():
            if len(positions) > 1:
                errors.update(positions)

    return [list(p) for p in sorted(errors)]


def rotate(image, turns):
    return np.ascontiguousarray(np.rot90(image, -turns))


def pick_orientation(h4, sample_count=12):
    pages, spans, checks = [], [], []

    for turns in range(4):
        found = candidate_cells(rotate(h4, turns))

        indices = np.linspace(
            0,
            len(found) - 1,
            min(sample_count, len(found)),
            dtype=int,
        )

        start = len(pages)
        pages.extend(
            glyph_variant(found[i][2], 28) for i in indices
        )
        spans.append((start, len(pages)))

        checks.append(
            {
                "angle": turns * 90,
                "filled_samples": len(indices),
            }
        )

    readings = ocr_pages(pages)

    for check, (start, end) in zip(checks, spans):
        values = readings[start:end]
        check["recognized"] = sum(d > 0 for d, _ in values)
        check["score"] = sum(
            (0.5 + max(0, confidence) / 100) if digit else 0
            for digit, confidence in values
        ) / max(1, len(values))

    best = max(range(4), key=lambda i: checks[i]["score"])
    return rotate(h4, best), best * 90, checks


def recognize(
    image,
    prepared=True,
    retry=8,
    details=False,
    ensemble=True,
):
    # No constraints, solver, annotations or other photos influence OCR.
    found = candidate_cells(image, prepared)
    votes = [dict() for _ in found]
    traces = [[] for _ in found]

    variants = (
        [
            (size, psm)
            for size in (28, 36, 44, 52)
            for psm in (10, 13)
        ]
        if prepared and ensemble
        else [(28, 10)] if prepared else [(None, 10)]
    )

    for size, psm in variants:
        pages = [
            glyph_variant(glyph, size) if size else glyph
            for _, _, glyph in found
        ]

        for i, (digit, confidence) in enumerate(
            ocr_pages(pages, psm)
        ):
            traces[i].append(
                {
                    "digit": digit,
                    "confidence": confidence,
                    "size": size,
                    "psm": psm,
                }
            )

            if digit:
                weight = (
                    1 + max(0, confidence) / 100
                ) * (2 if psm == 10 else 1)

                votes[i][digit] = (
                    votes[i].get(digit, 0) + weight
                )

    grid = [[0] * 9 for _ in range(9)]
    unread, records = [], []

    for i, (r, c, glyph) in enumerate(found):
        digit = max(votes[i], key=votes[i].get) if votes[i] else 0
        grid[r][c] = digit

        if digit == 0:
            unread.append([r, c])

        total = sum(votes[i].values())

        records.append(
            {
                "row": r,
                "col": c,
                "digit": digit,
                "vote_fraction": (
                    votes[i].get(digit, 0) / total
                    if total else 0
                ),
                "readings": traces[i],
            }
        )

    return (
        (grid, unread, records)
        if details else (grid, unread)
    )


def save_debug(image, grid, unread, conflicts, filename):
    output = image.copy()
    unread_set = {tuple(p) for p in unread}

    for r in range(9):
        for c in range(9):
            digit = grid[r][c]

            if digit:
                label = str(digit)
            elif (r, c) in unread_set:
                label = "?"
            else:
                label = ""

            if label:
                color = (
                    (0, 180, 0) if digit else (0, 0, 255)
                )

                cv2.putText(
                    output,
                    label,
                    (c * 50 + 3, r * 50 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

    for p in range(0, 451, 50):
        x = min(449, p)
        cv2.line(
            output, (x, 0), (x, 449), (255, 170, 0), 1
        )
        cv2.line(
            output, (0, x), (449, x), (255, 170, 0), 1
        )

    for r, c in conflicts:
        cv2.rectangle(
            output,
            (c * 50 + 2, r * 50 + 2),
            ((c + 1) * 50 - 2, (r + 1) * 50 - 2),
            (0, 0, 255),
            2,
        )

    cv2.imwrite(str(filename), output)


def spacing_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    result, offsets, straight = {}, [], []

    for axis, label in [(0, "vertical"), (1, "horizontal")]:
        kernel = np.ones(
            (25, 1) if axis == 0 else (1, 25),
            np.uint8,
        )
        lines = (
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) > 0
        )
        profile = lines.mean(axis=axis)

        positions, deviations, missing = [], [], []

        for ideal in range(0, 451, 50):
            expected = min(449, ideal)
            lo = max(0, expected - 24)
            hi = min(450, expected + 25)
            p = lo + int(profile[lo:hi].argmax())

            if profile[p] < 0.10:
                positions.append(None)
                deviations.append(None)
                missing.append(ideal)
                continue

            positions.append(p)
            deviations.append(p - expected)
            offsets.append(abs(p - expected))

            local = (
                lines[:, max(0, p - 3):min(450, p + 4)]
                if axis == 0
                else lines[
                    max(0, p - 3):min(450, p + 4), :
                ].T
            )

            yy, xx = np.where(local)

            if len(yy) > 25:
                fit = np.polyfit(yy, xx, 1)
                straight.append(
                    float(
                        np.sqrt(
                            np.mean(
                                (xx - np.polyval(fit, yy)) ** 2
                            )
                        )
                    )
                )

        result[label] = {
            "positions": positions,
            "offsets_px": deviations,
            "missing_lines": missing,
        }

    result["spacing_mae_px"] = (
        float(np.mean(offsets)) if offsets else None
    )
    result["line_fit_rms_px"] = (
        float(np.mean(straight)) if straight else None
    )
    result["missing_line_count"] = sum(
        len(result[key]["missing_lines"])
        for key in ("vertical", "horizontal")
    )

    return result


def spacing_error(image):
    return spacing_metrics(image)["spacing_mae_px"]


def compare_grid(gt, prediction):
    a = np.asarray(gt)
    b = np.asarray(prediction)

    if a.shape != (9, 9) or b.shape != (9, 9):
        raise ValueError("Both grids must be 9x9")

    correct = int((a == b).sum())
    tp = int(((a == 0) & (b == 0)).sum())
    fp = int(((a != 0) & (b == 0)).sum())
    fn = int(((a == 0) & (b != 0)).sum())

    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 1.0

    return {
        "cell_accuracy": correct / 81,
        "empty_f1": f1,
        "exact_match": int(correct == 81),
    }


def load_annotation(folder, name):
    if folder is None:
        return None, None

    path = folder / f"{name}.json"
    if not path.exists():
        return None, None

    ann = json.loads(path.read_text(encoding="utf-8"))
    status = ann.get("annotation_status", {})
    grid = np.asarray(ann.get("grid", []))

    # Standard Appendix A annotations need no custom verified field.
    usable = status.get("grid") == "visually_verified" or (
        ann.get("verified") is not False
        and status.get("grid") not in (
            "needs_visual_verification",
            "unverified",
            "draft",
        )
    )

    if (
        not usable
        or grid.shape != (9, 9)
        or not np.issubdtype(grid.dtype, np.integer)
        or np.any((grid < 0) | (grid > 9))
    ):
        grid = None
    else:
        grid = grid.tolist()

    corners = np.asarray(
        ann.get("corners", []), dtype=float
    )

    if (
        corners.shape != (4, 2)
        or not np.isfinite(corners).all()
    ):
        corners = None

    return grid, corners


def write_csv(path, rows):
    if not rows:
        return

    fields = sorted(
        set().union(*(row.keys() for row in rows))
    )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)


def process_image(
    path,
    outdir,
    analyze=False,
    annotations=None,
    forced_rotation=None,
):
    started = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)
    name = path.stem

    result = {
        "image": path.name,
        "verdict": "not_found",
    }

    for suffix in (
        "grid.txt",
        "analysis.json",
        "debug.png",
        "cells.png",
        "rect_A3.png",
        "rect_A4.png",
        "rect_H4.png",
        "rect_H4_upright.png",
    ):
        (outdir / f"{name}_{suffix}").unlink(
            missing_ok=True
        )

    try:
        image = cv2.imread(str(path))

        if image is None:
            raise ValueError("Cannot read image")

        work, gray, ink = preprocess(image)
        response, points = harris(gray)
        small_corners = select_corners(ink, points)

        save_harris(
            work,
            response,
            points,
            small_corners,
            outdir / f"{name}_harris.png",
        )

        if small_corners is None:
            raise ValueError("Sudoku corners not detected")

        corners = small_corners * np.float32(
            [
                image.shape[1] / work.shape[1],
                image.shape[0] / work.shape[0],
            ]
        )

        warps, a3, a4, h4, e3, e4 = rectify(
            image, corners
        )

        for label, frame in warps.items():
            cv2.imwrite(
                str(outdir / f"{name}_rect_{label}.png"),
                frame,
            )

        if forced_rotation is None:
            upright, angle, checks = pick_orientation(
                warps["H4"]
            )
        else:
            angle, checks = forced_rotation, []
            upright = rotate(
                warps["H4"], angle // 90
            )

        cv2.imwrite(
            str(outdir / f"{name}_rect_H4_upright.png"),
            upright,
        )

        grid, unread, readings = recognize(
            upright, details=True
        )
        conflicts = conflicts_in(grid)

        save_debug(
            upright,
            grid,
            unread,
            conflicts,
            outdir / f"{name}_debug.png",
        )

        sheet = np.full((720, 720), 255, np.uint8)

        for r, c, glyph in candidate_cells(upright):
            sheet[
                r * 80:r * 80 + 80,
                c * 80:c * 80 + 80,
            ] = glyph_variant(glyph, 28)

        cv2.imwrite(
            str(outdir / f"{name}_cells.png"),
            sheet,
        )

        (outdir / f"{name}_grid.txt").write_text(
            "\n".join(
                "".join(map(str, row)) for row in grid
            ) + "\n",
            encoding="utf-8",
        )

        mapped = cv2.perspectiveTransform(
            corners.reshape(1, 4, 2), h4
        )[0]

        result.update(
            corners=corners.tolist(),
            A3=a3.tolist(),
            A4=a4.tolist(),
            H4=h4.tolist(),
            a3_corner_err_px=e3,
            a4_rms_px=e4,
            a3_unused_br_target=(
                a3 @ np.r_[corners[2], 1]
            ).tolist(),
            h4_max_residual_px=float(
                np.linalg.norm(
                    mapped - DST, axis=1
                ).max()
            ),
            rotation_clockwise_degrees=angle,
            orientation_checks=checks,
            grid=grid,
            conflicts=conflicts,
            unread_cells=unread,
            ocr_cells=readings,
            candidate_count=len(points),
            verdict="invalid" if conflicts else "valid",
            recognition_complete=not unread,
        )

        # Optional research experiments are outside core runtime.
        result["runtime_s"] = round(
            time.perf_counter() - started, 4
        )

        if analyze:
            analyze_image(
                path,
                outdir,
                annotations,
                corners,
                warps,
                angle,
                grid,
                e3,
                e4,
                len(points),
            )

    except Exception as exc:
        result["error"] = str(exc)

        if "grid" not in result:
            result["verdict"] = "not_found"

        print(f"[ERROR] {path.name}: {exc}")

    result.setdefault(
        "runtime_s",
        round(time.perf_counter() - started, 4),
    )
    result["total_runtime_s"] = round(
        time.perf_counter() - started, 4
    )

    (outdir / f"{name}_result.json").write_text(
        json.dumps(
            result, indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )

    print(
        f"[{result['verdict']}] {path.name} | "
        f"core={result['runtime_s']:.3f}s | "
        f"rotate={result.get('rotation_clockwise_degrees', '-')} | "
        f"unread={len(result.get('unread_cells', []))}",
        flush=True,
    )

    return result


def analyze_image(
    path,
    outdir,
    annotations,
    corners,
    warps,
    angle,
    grid,
    e3,
    e4,
    candidates,
):
    gt, gt_corners = load_annotation(
        annotations, path.stem
    )

    sides = np.linalg.norm(
        np.roll(corners, -1, axis=0) - corners,
        axis=1,
    )

    oblique = float(
        1 - min(
            min(sides[0], sides[2])
            / max(sides[0], sides[2]),
            min(sides[1], sides[3])
            / max(sides[1], sides[3]),
        )
    )

    analysis = {
        "image": path.name,
        "candidate_count": candidates,
        "A3_corner_error_px": e3,
        "A4_rms_px": e4,
        "obliqueness": oblique,
        "has_grid_reference": gt is not None,
        "has_corner_reference": gt_corners is not None,
        "rotation_clockwise_degrees": angle,
    }

    if gt_corners is not None:
        distances = np.linalg.norm(
            corners - gt_corners, axis=1
        )
        side = np.linalg.norm(
            np.roll(gt_corners, -1, axis=0)
            - gt_corners,
            axis=1,
        ).mean()

        analysis.update(
            corner_error_px=float(distances.mean()),
            corner_error_pct=float(
                100 * distances.mean() / side
            ),
        )

    confusions = []

    for label, frame in warps.items():
        aligned = rotate(frame, angle // 90)
        spacing = spacing_metrics(frame)

        (
            outdir / f"{path.stem}_{label}_lines.json"
        ).write_text(
            json.dumps(spacing, indent=2),
            encoding="utf-8",
        )

        for key in (
            "spacing_mae_px",
            "line_fit_rms_px",
            "missing_line_count",
        ):
            analysis[f"{label}_{key}"] = spacing[key]

        if gt is not None:
            prediction = (
                grid
                if label == "H4"
                else recognize(aligned)[0]
            )

            for key, value in compare_grid(
                gt, prediction
            ).items():
                analysis[f"{label}_{key}"] = value

            if label == "H4":
                errors = np.argwhere(
                    np.asarray(gt) != np.asarray(grid)
                )

                for r, c in errors:
                    r, c = int(r), int(c)

                    confusions.append(
                        {
                            "row": r,
                            "col": c,
                            "true": gt[r][c],
                            "predicted": grid[r][c],
                        }
                    )

                    folder = outdir / "confusion_crops"
                    folder.mkdir(exist_ok=True)

                    filename = (
                        f"{path.stem}_r{r+1}c{c+1}"
                        f"_true{gt[r][c]}"
                        f"_pred{grid[r][c]}.png"
                    )

                    cv2.imwrite(
                        str(folder / filename),
                        aligned[
                            r * 50:(r + 1) * 50,
                            c * 50:(c + 1) * 50,
                        ],
                    )

    if gt is not None:
        upright = rotate(warps["H4"], angle // 90)

        raw_grid = recognize(
            upright, prepared=False
        )[0]
        single_grid = recognize(
            upright, ensemble=False
        )[0]

        analysis["H4_raw_cell_accuracy"] = compare_grid(
            gt, raw_grid
        )["cell_accuracy"]

        analysis[
            "H4_prepared_single_cell_accuracy"
        ] = compare_grid(
            gt, single_grid
        )["cell_accuracy"]

        analysis["H4_preparation_gain"] = (
            analysis["H4_prepared_single_cell_accuracy"]
            - analysis["H4_raw_cell_accuracy"]
        )

        analysis["H4_ensemble_gain"] = (
            analysis["H4_cell_accuracy"]
            - analysis["H4_prepared_single_cell_accuracy"]
        )

        for label, prediction in [
            ("raw", raw_grid),
            ("prepared", grid),
        ]:
            analysis[f"H4_{label}_confusions"] = [
                {
                    "row": int(r),
                    "col": int(c),
                    "true": gt[r][c],
                    "predicted": prediction[r][c],
                }
                for r, c in np.argwhere(
                    np.asarray(gt)
                    != np.asarray(prediction)
                )
            ]

        folder = outdir / "confusion_crops"
        folder.mkdir(exist_ok=True)

        for error in analysis["H4_raw_confusions"]:
            r, c = error["row"], error["col"]

            cell = upright[
                r * 50:(r + 1) * 50,
                c * 50:(c + 1) * 50,
            ]

            raw = cv2.resize(
                cv2.cvtColor(
                    cell, cv2.COLOR_BGR2GRAY
                )[8:42, 8:42],
                (80, 80),
            )

            glyph = cell_glyph(cell)

            if glyph is None:
                glyph = np.full(
                    (80, 80), 255, np.uint8
                )

            example = cv2.hconcat(
                [raw, glyph_variant(glyph, 28)]
            )

            filename = (
                f"{path.stem}_raw_r{r+1}c{c+1}"
                f"_true{error['true']}"
                f"_pred{error['predicted']}.png"
            )

            cv2.imwrite(
                str(folder / filename), example
            )

    analysis["confusions"] = confusions

    (outdir / f"{path.stem}_analysis.json").write_text(
        json.dumps(analysis, indent=2),
        encoding="utf-8",
    )

    tiles = []

    for label, frame in warps.items():
        tile = cv2.copyMakeBorder(
            frame,
            30,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(255,) * 3,
        )

        cv2.putText(
            tile,
            label,
            (15, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        tiles.append(tile)

    cv2.imwrite(
        str(outdir / f"{path.stem}_comparison.png"),
        cv2.hconcat(tiles),
    )


def harris_study(paths, outdir, annotations):
    settings = [
        (2, 3, 0.04, 0.002),
        (3, 3, 0.04, 0.002),
        (2, 5, 0.04, 0.002),
        (2, 3, 0.06, 0.002),
        (2, 3, 0.04, 0.01),
    ]

    rows = []

    for path in paths[:6]:
        image = cv2.imread(str(path))

        if image is None:
            continue

        work, gray, ink = preprocess(image)
        _, gt = load_annotation(
            annotations, path.stem
        )

        for block, aperture, k, threshold in settings:
            _, points = harris(
                gray, block, aperture, k, threshold
            )
            corners = select_corners(ink, points)

            row = {
                "image": path.name,
                "blockSize": block,
                "ksize": aperture,
                "k": k,
                "threshold": threshold,
                "candidate_count": len(points),
                "found": corners is not None,
                "corner_error_pct": "",
            }

            if corners is not None and gt is not None:
                scale = np.array(
                    [
                        image.shape[1] / work.shape[1],
                        image.shape[0] / work.shape[0],
                    ]
                )

                side = np.linalg.norm(
                    np.roll(gt, -1, axis=0) - gt,
                    axis=1,
                ).mean()

                error = np.linalg.norm(
                    corners * scale - gt,
                    axis=1,
                ).mean()

                row["corner_error_pct"] = float(
                    error / side * 100
                )

            rows.append(row)

    write_csv(
        outdir / "harris_parameter_study.csv",
        rows,
    )


def make_plots(outdir, rows):
    if not rows:
        return

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    x = [r["obliqueness"] for r in rows]
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4)
    )

    for key, label in [
        ("A3_corner_error_px", "A3 fourth corner"),
        ("A4_rms_px", "A4 RMS"),
    ]:
        axes[0].scatter(
            x,
            [r[key] for r in rows],
            label=label,
        )

    for model in ("A3", "A4", "H4"):
        usable = [
            r for r in rows
            if r.get(f"{model}_spacing_mae_px")
            is not None
        ]

        axes[1].scatter(
            [r["obliqueness"] for r in usable],
            [
                r[f"{model}_spacing_mae_px"]
                for r in usable
            ],
            label=model,
        )

        usable = [
            r for r in rows
            if f"{model}_cell_accuracy" in r
        ]

        if usable:
            axes[2].scatter(
                [
                    r["obliqueness"]
                    for r in usable
                ],
                [
                    r[f"{model}_cell_accuracy"]
                    for r in usable
                ],
                label=model,
            )

    for ax, ylabel in zip(
        axes,
        (
            "Corner residual (px)",
            "Grid position MAE (px)",
            "Cell accuracy",
        ),
    ):
        ax.set(
            xlabel=(
                "Obliqueness: 1 - minimum "
                "opposite-side ratio"
            ),
            ylabel=ylabel,
        )

        ax.grid(alpha=0.25)

        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    fig.tight_layout()
    fig.savefig(
        outdir / "stage4_plots.png", dpi=160
    )
    plt.close(fig)


def annotate_reference_corners(paths, folder):
    # Ground-truth UI only; automatic scanning never calls this.
    folder.mkdir(parents=True, exist_ok=True)
    labels = ("TL", "TR", "BR", "BL")

    for path in paths:
        image = cv2.imread(str(path))

        if image is None:
            print("Cannot read:", path)
            continue

        scale = min(
            1.0,
            1100 / image.shape[1],
            800 / image.shape[0],
        )

        display = cv2.resize(
            image, None, fx=scale, fy=scale
        )

        target = folder / f"{path.stem}.json"

        annotation = (
            json.loads(
                target.read_text(encoding="utf-8")
            )
            if target.exists()
            else {
                "image": path.name,
                "conditions": [],
            }
        )

        points = []
        window = f"Reference corners: {path.name}"

        def mouse(event, x, y, flags, param):
            if (
                event == cv2.EVENT_LBUTTONDOWN
                and len(points) < 4
            ):
                points.append(
                    [x / scale, y / scale]
                )

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, mouse)

        print(
            path.name,
            ": click image TL, TR, BR, BL; "
            "S save, R reset, Esc skip.",
        )

        while True:
            canvas = display.copy()

            for i, (x, y) in enumerate(points):
                xy = (
                    round(x * scale),
                    round(y * scale),
                )

                cv2.circle(
                    canvas, xy, 6, (0, 0, 255), -1
                )

                cv2.putText(
                    canvas,
                    labels[i],
                    (xy[0] + 8, xy[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            next_label = (
                labels[len(points)]
                if len(points) < 4
                else "S: save / R: reset"
            )

            cv2.putText(
                canvas,
                next_label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 150, 0),
                2,
            )

            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 255

            if key == 27:
                break

            if key in (ord("r"), ord("R")):
                points.clear()

            if (
                key in (ord("s"), ord("S"))
                and len(points) == 4
            ):
                quad = np.float32(points).reshape(
                    4, 1, 2
                )

                if (
                    not cv2.isContourConvex(quad)
                    or cv2.contourArea(quad) < 100
                ):
                    print(
                        "Invalid corner order; reset "
                        "and click TL, TR, BR, BL."
                    )
                    continue

                annotation["corners"] = points

                annotation.setdefault(
                    "annotation_status", {}
                )["corners"] = "manually_annotated"

                annotation["corner_reference_source"] = (
                    "Independent manual clicks "
                    "on original photograph."
                )

                target.write_text(
                    json.dumps(
                        annotation, indent=2
                    ),
                    encoding="utf-8",
                )

                print("Saved:", target)
                break

        cv2.destroyWindow(window)


def evaluation_summary(paths, outdir, annotations):
    results = [
        json.loads(
            (
                outdir / f"{p.stem}_result.json"
            ).read_text(encoding="utf-8")
        )
        for p in paths
    ]

    known = 0
    correct = 0
    exact = 0
    empty_tp = 0
    empty_fp = 0
    empty_fn = 0
    confusions = {}

    for path, result in zip(paths, results):
        gt, _ = load_annotation(
            annotations, path.stem
        )

        if gt is None:
            continue

        known += 1
        a = np.asarray(gt)

        if "grid" in result:
            b = np.asarray(result["grid"])

            correct += int((a == b).sum())
            exact += int(np.array_equal(a, b))

            empty_tp += int(
                ((a == 0) & (b == 0)).sum()
            )
            empty_fp += int(
                ((a != 0) & (b == 0)).sum()
            )
            empty_fn += int(
                ((a == 0) & (b != 0)).sum()
            )

            for truth, prediction in zip(
                a[a != b], b[a != b]
            ):
                key = f"{truth}->{prediction}"
                confusions[key] = (
                    confusions.get(key, 0) + 1
                )
        else:
            # Failed scans earn zero cell/exact credit.
            empty_fn += int((a == 0).sum())

    denominator = (
        2 * empty_tp + empty_fp + empty_fn
    )

    summary = {
        "image_count": len(paths),
        "grid_reference_count": known,
        "cell_accuracy": (
            correct / (81 * known)
            if known else None
        ),
        "correct_cells": correct,
        "total_reference_cells": known * 81,
        "grid_exact_match": (
            exact / known if known else None
        ),
        "empty_f1": (
            2 * empty_tp / denominator
            if denominator
            else (1.0 if known else None)
        ),
        "robustness": sum(
            "grid" in r and r["runtime_s"] <= 20
            for r in results
        ) / len(results),
        "mean_runtime_s": float(
            np.mean(
                [r["runtime_s"] for r in results]
            )
        ),
        "max_runtime_s": max(
            r["runtime_s"] for r in results
        ),
        "confusions": dict(
            sorted(
                confusions.items(),
                key=lambda item: -item[1],
            )
        ),
        "corner_reference_count": sum(
            load_annotation(
                annotations, p.stem
            )[1] is not None
            for p in paths
        ),
        "evaluation_scope": (
            "Development photos only; "
            "no hidden-set claim."
        ),
    }

    (
        outdir / "evaluation_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="COMP482 SudokuScan"
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )
    group.add_argument("--input", type=Path)
    group.add_argument("--batch", type=Path)

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("out"),
    )
    parser.add_argument(
        "--annotations", type=Path
    )
    parser.add_argument(
        "--analyze", action="store_true"
    )
    parser.add_argument(
        "--harris-study", action="store_true"
    )
    parser.add_argument(
        "--rotation",
        type=int,
        choices=[0, 90, 180, 270],
        default=None,
    )
    parser.add_argument(
        "--tesseract",
        help=(
            "Full path to tesseract.exe "
            "if absent from PATH"
        ),
    )
    parser.add_argument(
        "--annotate-corners",
        action="store_true",
        help=(
            "Manual ground-truth annotation UI; "
            "never used by scanning"
        ),
    )

    args = parser.parse_args()

    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = (
            args.tesseract
        )
    elif not shutil.which("tesseract"):
        windows = Path(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

        if windows.exists():
            pytesseract.pytesseract.tesseract_cmd = (
                str(windows)
            )

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        parser.error(
            "Tesseract is not available. "
            "Install it or pass --tesseract: "
            f"{exc}"
        )

    args.outdir.mkdir(
        parents=True, exist_ok=True
    )

    if args.input:
        paths = [args.input]
    else:
        paths = sorted(
            p for p in args.batch.iterdir()
            if p.suffix.lower()
            in (".jpg", ".jpeg", ".png")
        )

    if not paths:
        parser.error("No images found")

    if args.annotate_corners:
        if args.annotations is None:
            parser.error(
                "--annotate-corners requires "
                "--annotations"
            )

        annotate_reference_corners(
            paths, args.annotations
        )
        return

    for path in paths:
        process_image(
            path,
            args.outdir,
            args.analyze,
            args.annotations,
            args.rotation,
        )

    if args.analyze:
        rows = []

        for path in paths:
            analysis = (
                args.outdir
                / f"{path.stem}_analysis.json"
            )

            if analysis.exists():
                rows.append(
                    json.loads(
                        analysis.read_text(
                            encoding="utf-8"
                        )
                    )
                )

        csv_rows = [
            {
                key: value
                for key, value in row.items()
                if not isinstance(
                    value, (list, dict)
                )
            }
            for row in rows
        ]

        write_csv(
            args.outdir / "stage4_comparison.csv",
            csv_rows,
        )
        make_plots(args.outdir, rows)
        evaluation_summary(
            paths,
            args.outdir,
            args.annotations,
        )

    if args.harris_study:
        harris_study(
            paths,
            args.outdir,
            args.annotations,
        )


if __name__ == "__main__":
    main()