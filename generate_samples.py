"""
Generate sample photos and matching sketches so PhotoSketch is demoable
out of the box, without needing the user to supply their own images.

Creates simple synthetic "face" illustrations (distinct per identity) as the
photo gallery, and a corresponding pencil-sketch for each identity that can be
used as a query. Everything is deterministic so results are reproducible.

Run:  python generate_samples.py
"""

import os
import cv2
import numpy as np

import engine

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
PHOTOS_DIR = os.path.join(SAMPLES_DIR, "photos")
SKETCHES_DIR = os.path.join(SAMPLES_DIR, "sketches")


# Each identity has a distinct face geometry so matches are meaningful.
IDENTITIES = [
    # name, skin(BGR), face_w, face_h, eye_dx, eye_y, nose_len, mouth_w, hair(BGR)
    ("person_1", (170, 190, 215), 150, 195, 42, 150, 46, 66, (40, 40, 45)),
    ("person_2", (150, 175, 205), 172, 172, 52, 140, 38, 84, (30, 55, 95)),
    ("person_3", (160, 185, 210), 138, 205, 36, 160, 54, 58, (60, 60, 60)),
    ("person_4", (175, 195, 220), 185, 160, 58, 132, 34, 92, (20, 20, 25)),
    ("person_5", (155, 180, 208), 158, 188, 46, 152, 44, 72, (48, 72, 110)),
]

CANVAS = 320


def draw_face(spec) -> np.ndarray:
    (name, skin, fw, fh, eye_dx, eye_y, nose_len, mouth_w, hair) = spec
    img = np.full((CANVAS, CANVAS, 3), 245, dtype=np.uint8)
    cx, cy = CANVAS // 2, CANVAS // 2 + 8

    # Hair (drawn behind face as a larger ellipse up top)
    cv2.ellipse(img, (cx, cy - 20), (fw // 2 + 12, fh // 2 + 6),
                0, 180, 360, hair, -1)

    # Face oval
    cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, skin, -1)
    cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 0, 360,
                (120, 120, 130), 2)

    # Eyes (whites + iris)
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        cv2.ellipse(img, (ex, eye_y), (20, 12), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (ex, eye_y), 7, (60, 40, 30), -1)
        cv2.circle(img, (ex, eye_y), 3, (10, 10, 10), -1)
        cv2.ellipse(img, (ex, eye_y), (20, 12), 0, 0, 360, (90, 90, 90), 1)
        # Eyebrow
        cv2.line(img, (ex - 18, eye_y - 20), (ex + 18, eye_y - 24),
                 hair, 3)

    # Nose
    nose_top = eye_y + 14
    cv2.line(img, (cx, nose_top), (cx - 10, nose_top + nose_len),
             (120, 110, 120), 2)
    cv2.line(img, (cx - 10, nose_top + nose_len), (cx + 8, nose_top + nose_len),
             (120, 110, 120), 2)

    # Mouth
    mouth_y = nose_top + nose_len + 34
    cv2.ellipse(img, (cx, mouth_y), (mouth_w // 2, 16), 0, 10, 170,
                (70, 70, 130), 3)

    return img


def main():
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    os.makedirs(SKETCHES_DIR, exist_ok=True)

    for spec in IDENTITIES:
        name = spec[0]
        photo = draw_face(spec)

        photo_path = os.path.join(PHOTOS_DIR, f"{name}.png")
        cv2.imwrite(photo_path, photo)

        # Build a query sketch from the same face (sketch domain).
        sketch = engine.photo_to_sketch(photo)
        sketch_path = os.path.join(SKETCHES_DIR, f"{name}_sketch.png")
        cv2.imwrite(sketch_path, sketch)

        print(f"Created {photo_path} and {sketch_path}")

    print(f"\nDone. {len(IDENTITIES)} photos in {PHOTOS_DIR}")
    print(f"      {len(IDENTITIES)} sketches in {SKETCHES_DIR}")


if __name__ == "__main__":
    main()
