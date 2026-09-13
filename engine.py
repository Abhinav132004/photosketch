"""
PhotoSketch - Core Matching Engine
-----------------------------------
Classical computer-vision pipeline for matching hand-drawn sketches to photographs.

The modality gap between a line sketch and a real photo is bridged by:
  1. Preprocessing      -> common size, grayscale, denoise
  2. Sketch conversion  -> map a photo into the "sketch domain"
  3. Feature extraction -> Canny edges, ORB keypoints
  4. Similarity scoring -> SSIM + ORB match ratio + edge agreement, fused
  5. Ranking            -> sort candidate photos by combined score

No model training required. Runs on ordinary hardware.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# Standard working size for all comparisons (square keeps SSIM windows valid)
STANDARD_SIZE = (256, 256)


# --------------------------------------------------------------------------- #
# 1. PREPROCESSING
# --------------------------------------------------------------------------- #
def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel grayscale image regardless of input channels."""
    if image is None:
        raise ValueError("Received an empty image.")
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preprocess(image: np.ndarray, size: tuple[int, int] = STANDARD_SIZE) -> np.ndarray:
    """Resize to a common size, grayscale, denoise and equalise contrast."""
    gray = to_gray(image)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    # Light denoise so stray marks / scan artefacts don't dominate comparison
    denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=50, sigmaSpace=50)
    # Normalise lighting differences between sketch and photo
    equalised = cv2.equalizeHist(denoised)
    return equalised


# --------------------------------------------------------------------------- #
# 2. SKETCH-DOMAIN CONVERSION (photo -> pencil sketch)
# --------------------------------------------------------------------------- #
def photo_to_sketch(image: np.ndarray, blur_ksize: int = 21) -> np.ndarray:
    """
    Convert a photograph into a pencil-sketch image using the classic
    grayscale -> invert -> blur -> colour-dodge technique.
    Returns a single-channel (grayscale) sketch.
    """
    gray = to_gray(image)
    inverted = 255 - gray
    # Kernel must be odd
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(inverted, (blur_ksize, blur_ksize), sigmaX=0)
    inverted_blur = 255 - blurred
    # Colour dodge blend
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return sketch


# --------------------------------------------------------------------------- #
# 3. FEATURE EXTRACTION
# --------------------------------------------------------------------------- #
def edge_map(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge map - the structural signature shared by sketch and photo."""
    return cv2.Canny(gray, low, high)


_ORB = cv2.ORB_create(nfeatures=500)


def orb_features(gray: np.ndarray):
    """Detect ORB keypoints and compute binary descriptors."""
    keypoints, descriptors = _ORB.detectAndCompute(gray, None)
    return keypoints, descriptors


# --------------------------------------------------------------------------- #
# 4. SIMILARITY MEASURES
# --------------------------------------------------------------------------- #
def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity Index in [0, 1] (clamped from [-1, 1])."""
    score = ssim(a, b)
    return float(max(0.0, score))


def orb_match_score(desc_a, desc_b) -> float:
    """
    Ratio of 'good' ORB matches to the number of query descriptors, in [0, 1].
    Uses Hamming distance (binary descriptors) with Lowe's ratio test.
    """
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        knn = bf.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return 0.0

    good = 0
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:  # Lowe's ratio test
            good += 1

    return float(min(1.0, good / max(1, len(desc_a))))


def edge_agreement(edges_a: np.ndarray, edges_b: np.ndarray) -> float:
    """
    Intersection-over-union of the two binary edge maps, in [0, 1].
    Measures how well facial contours line up.
    """
    a = edges_a > 0
    b = edges_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    return float(intersection / union)


# --------------------------------------------------------------------------- #
# 5. COMBINED SCORE + RANKING
# --------------------------------------------------------------------------- #
# Weights for fusing the three complementary measures. Tuned so global
# structure (SSIM) leads, with local landmarks (ORB) and contours (edges)
# providing supporting evidence.
WEIGHTS = {"ssim": 0.5, "orb": 0.3, "edge": 0.2}


def _prepare(image: np.ndarray) -> dict:
    """Precompute everything needed to compare one image."""
    pre = preprocess(image)
    sketch_domain = photo_to_sketch(cv2.resize(to_gray(image), STANDARD_SIZE))
    edges = edge_map(pre)
    _, desc = orb_features(pre)
    return {
        "pre": pre,
        "sketch_domain": sketch_domain,
        "edges": edges,
        "desc": desc,
    }


def compare(sketch_img: np.ndarray, photo_img: np.ndarray) -> dict:
    """
    Compare one sketch against one photo. The photo is mapped into the sketch
    domain first so both are represented consistently.
    Returns individual measures and the fused score (all in [0, 1]).
    """
    # Sketch: use its preprocessed form directly (it is already a sketch)
    s_pre = preprocess(sketch_img)
    s_edges = edge_map(s_pre)
    _, s_desc = orb_features(s_pre)

    # Photo: convert to sketch domain, then preprocess that
    p_sketch = photo_to_sketch(photo_img)
    p_pre = preprocess(p_sketch)
    p_edges = edge_map(p_pre)
    _, p_desc = orb_features(p_pre)

    s_ssim = ssim_score(s_pre, p_pre)
    s_orb = orb_match_score(s_desc, p_desc)
    s_edge = edge_agreement(s_edges, p_edges)

    combined = (
        WEIGHTS["ssim"] * s_ssim
        + WEIGHTS["orb"] * s_orb
        + WEIGHTS["edge"] * s_edge
    )

    return {
        "ssim": round(s_ssim * 100, 2),
        "orb": round(s_orb * 100, 2),
        "edge": round(s_edge * 100, 2),
        "score": round(combined * 100, 2),
    }


def rank_matches(sketch_img: np.ndarray, photos: list[dict]) -> list[dict]:
    """
    Rank candidate photos against a sketch.
    `photos` is a list of {"name": str, "image": np.ndarray}.
    Returns the same dicts enriched with score fields, sorted best-first.
    """
    results = []
    for item in photos:
        scores = compare(sketch_img, item["image"])
        results.append({**item, **scores})

    results.sort(key=lambda r: r["score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
    return results


def risk_free_label(score: float) -> tuple[str, str]:
    """Map a combined score (0-100) to a confidence label and colour."""
    if score >= 55:
        return "Strong Match", "#16a34a"
    if score >= 40:
        return "Likely Match", "#65a30d"
    if score >= 28:
        return "Possible Match", "#d97706"
    return "Weak Match", "#dc2626"
