"""
Download real face photos to replace the synthetic drawn samples, and
regenerate matching sketches + realistic hand-drawn query sketches.

Source: randomuser.me portrait set — free-to-use portrait photos served as
direct JPEGs. Good, clean faces for a demo gallery.

Run:  python download_real_faces.py
"""

import os
import time
import urllib.request
import ssl

import cv2
import numpy as np

import engine
import generate_test_queries  # reuse hand_drawn() for realistic queries

BASE = os.path.dirname(__file__)
PHOTOS_DIR = os.path.join(BASE, "samples", "photos")
SKETCHES_DIR = os.path.join(BASE, "samples", "sketches")
QUERIES_DIR = os.path.join(BASE, "test_queries")

# A curated spread of distinct portrait faces from randomuser.me
FACE_URLS = [
    "https://randomuser.me/api/portraits/men/32.jpg",
    "https://randomuser.me/api/portraits/women/44.jpg",
    "https://randomuser.me/api/portraits/men/75.jpg",
    "https://randomuser.me/api/portraits/women/68.jpg",
    "https://randomuser.me/api/portraits/men/12.jpg",
]
TARGET = 384  # output size (square)


def fetch(url: str) -> np.ndarray | None:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = resp.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        print(f"  download failed for {url}: {exc}")
        return None


def clean_dir(path):
    os.makedirs(path, exist_ok=True)
    for f in os.listdir(path):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(path, f))


def main():
    print("Downloading real face photos...")
    clean_dir(PHOTOS_DIR)
    clean_dir(SKETCHES_DIR)
    clean_dir(QUERIES_DIR)

    saved = 0
    for i, url in enumerate(FACE_URLS):
        img = fetch(url)
        if img is None:
            continue
        img = cv2.resize(img, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
        name = f"person_{saved + 1}"

        cv2.imwrite(os.path.join(PHOTOS_DIR, f"{name}.png"), img)
        sketch = engine.photo_to_sketch(img)
        cv2.imwrite(os.path.join(SKETCHES_DIR, f"{name}_sketch.png"), sketch)
        query = generate_test_queries.hand_drawn(sketch, seed=200 + saved)
        cv2.imwrite(os.path.join(QUERIES_DIR, f"{name}_query.png"), query)

        saved += 1
        print(f"  saved {name} ({saved}/{len(FACE_URLS)})")
        time.sleep(0.5)

    if saved == 0:
        print("\nNo faces downloaded (no internet?). "
              "Run generate_samples.py for synthetic fallback.")
    else:
        print(f"\nDone. {saved} real faces + sketches + queries created.")


if __name__ == "__main__":
    main()
