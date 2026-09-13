# PhotoSketch — AI-Based Photo and Sketch Matching System

Match hand-drawn facial sketches to real photographs using classical computer
vision. A sketch and a photo of the same person look very different pixel-wise,
so PhotoSketch compares **structure** (edges, keypoints, layout) instead of raw
pixels. No model training and no GPU required.

## Features

- **Sketch → Photo Matching** — upload a sketch and a gallery of photos, get a
  ranked list of matches with confidence scores.
- **Photo → Sketch** — convert any photograph into a pencil sketch and inspect
  the Canny edge map; download the sketch as PNG.
- **Explainable scores** — every match shows its SSIM, ORB and edge-agreement
  components, so you can see *why* a photo scored the way it did.
- **Works out of the box** — 5 sample photos + sketches are bundled.
- **Polished UI** — dark-themed Streamlit interface.

## How it works

```
Input sketch + candidate photos
        │
        ▼  Preprocess (resize 256×256, grayscale, denoise, equalise)
        ▼  Convert each photo → sketch domain (grayscale→invert→blur→dodge)
        ▼  Extract features: Canny edges + ORB keypoints
        ▼  Similarity: SSIM (0.5) + ORB match ratio (0.3) + edge IoU (0.2)
        ▼  Fuse into one score, rank photos best-first
   Ranked matches
```

## Project structure

```
PhotoSketch/
├── app.py               # Streamlit application (UI, 4 pages)
├── engine.py            # Core matching engine (CV pipeline)
├── generate_samples.py  # Creates sample photos + sketches
├── requirements.txt     # Dependencies
├── samples/
│   ├── photos/          # Sample gallery photos
│   └── sketches/        # Sample query sketches
└── venv/                # Virtual environment (created on setup)
```

## Setup (one time)

```powershell
# From the PhotoSketch folder
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe generate_samples.py
```

## Run

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Technology stack

Python · OpenCV · NumPy · scikit-image · Pillow · Streamlit

## Limitations & future scope

This is a classical, training-free system built for clarity and reproducibility.
It works best on clean, well-proportioned sketches and is intended as a
**shortlisting aid**, not a definitive identifier. Accuracy on rough real-world
forensic sketches would improve with a trained deep-learning model (e.g. a
Siamese CNN trained on the CUFS/CUFSF datasets) — a natural next step.
