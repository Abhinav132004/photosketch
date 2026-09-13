"""
Generate realistic 'hand-drawn style' query sketches for testing the
Sketch -> Photo matching feature.

Unlike samples/sketches (which are exact sketch conversions of the photos and
therefore match at 100%), these test queries are deliberately imperfect:
  - slight geometric warp (as if drawn by hand)
  - light random strokes / jitter
  - softened lines

This shows the matcher finding the correct person even when the sketch is NOT
a pixel-perfect copy of the photo - a far more convincing demo.

Run:  python generate_test_queries.py
Output: test_queries/<name>_query.png
"""

import os
import glob
import cv2
import numpy as np

import engine

PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "samples", "photos")
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_queries")


def hand_drawn(sketch: np.ndarray, seed: int) -> np.ndarray:
    """Make a clean sketch look hand-drawn: warp + jitter + soften."""
    rng = np.random.default_rng(seed)
    h, w = sketch.shape[:2]

    # 1) Gentle sinusoidal warp (wobbly hand-drawn lines)
    amp = 3.5
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xs + amp * np.sin(ys / 22.0 + rng.uniform(0, 3))).astype(np.float32)
    map_y = (ys + amp * np.cos(xs / 22.0 + rng.uniform(0, 3))).astype(np.float32)
    warped = cv2.remap(sketch, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

    # 2) Add a few faint stray pencil strokes
    canvas = warped.copy()
    for _ in range(rng.integers(4, 8)):
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        x2 = int(np.clip(x1 + rng.integers(-40, 40), 0, w - 1))
        y2 = int(np.clip(y1 + rng.integers(-40, 40), 0, h - 1))
        cv2.line(canvas, (x1, y1), (x2, y2), int(rng.integers(150, 210)), 1)

    # 3) Soften slightly so it reads like graphite, not a clean render
    softened = cv2.GaussianBlur(canvas, (3, 3), 0)
    return softened


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    photos = sorted(glob.glob(os.path.join(PHOTOS_DIR, "*.png")))
    if not photos:
        print("No sample photos found. Run generate_samples.py first.")
        return

    for i, path in enumerate(photos):
        name = os.path.splitext(os.path.basename(path))[0]
        photo = cv2.imread(path)
        clean_sketch = engine.photo_to_sketch(photo)
        query = hand_drawn(clean_sketch, seed=100 + i)
        out_path = os.path.join(OUT_DIR, f"{name}_query.png")
        cv2.imwrite(out_path, query)
        print(f"Created {out_path}")

    print(f"\nDone. {len(photos)} hand-drawn style query sketches in {OUT_DIR}")


if __name__ == "__main__":
    main()
