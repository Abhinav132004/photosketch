"""
PhotoSketch - AI-Based Photo and Sketch Matching System
Streamlit application with an animated, aesthetic single-screen UI.

Run:  streamlit run app.py
"""

import os
import glob

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import engine

# On a fresh deployment (e.g. Streamlit Cloud) the sample images may not exist.
# Generate them once at startup so the app is demoable out of the box.
def _ensure_samples():
    photos_dir = os.path.join(os.path.dirname(__file__), "samples", "photos")
    if not (os.path.isdir(photos_dir) and glob.glob(os.path.join(photos_dir, "*.png"))):
        try:
            import generate_samples
            generate_samples.main()
        except Exception as exc:  # pragma: no cover - best effort bootstrap
            print(f"Sample generation skipped: {exc}")

_ensure_samples()

# --------------------------------------------------------------------------- #
# PAGE CONFIG
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="PhotoSketch - AI Sketch Matching",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# GLOBAL STYLES + ANIMATIONS
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    /* ---- Animated aurora background ---- */
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(139,92,246,0.35), transparent 60%),
            radial-gradient(1000px 500px at 100% 0%, rgba(236,72,153,0.28), transparent 55%),
            radial-gradient(900px 700px at 50% 120%, rgba(56,189,248,0.25), transparent 60%),
            linear-gradient(135deg, #0b1020 0%, #0f172a 50%, #14082b 100%);
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed; inset: 0; z-index: 0; pointer-events: none;
        background:
            radial-gradient(600px 300px at 20% 30%, rgba(99,102,241,0.20), transparent 60%),
            radial-gradient(500px 260px at 80% 60%, rgba(236,72,153,0.18), transparent 60%);
        animation: floatBg 16s ease-in-out infinite alternate;
    }
    @keyframes floatBg {
        0%   { transform: translate3d(0,0,0) scale(1); opacity: .85; }
        100% { transform: translate3d(0,-30px,0) scale(1.08); opacity: 1; }
    }

    .block-container { padding-top: 1.2rem; max-width: 1250px; position: relative; z-index: 1; }
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    h1,h2,h3,h4 { font-family:'Space Grotesk','Poppins',sans-serif; }
    h1,h2,h3,h4,p,label,span,div,li { color:#e5e9f5; }
    #MainMenu, footer, header { visibility: hidden; }

    /* ---- Hero ---- */
    .hero {
        position: relative; overflow: hidden;
        border-radius: 26px; padding: 2.6rem 2.4rem; margin-bottom: 1.4rem;
        background: linear-gradient(120deg, rgba(99,102,241,.95), rgba(139,92,246,.9) 45%, rgba(236,72,153,.9));
        box-shadow: 0 24px 60px rgba(124,58,237,.45), inset 0 0 40px rgba(255,255,255,.06);
        animation: heroIn .9s cubic-bezier(.2,.8,.2,1) both;
    }
    .hero::after {
        content:""; position:absolute; top:-50%; left:-50%; width:200%; height:200%;
        background: conic-gradient(from 0deg, transparent 0 70%, rgba(255,255,255,.18) 85%, transparent 100%);
        animation: spin 9s linear infinite; opacity:.5;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes heroIn { from { opacity:0; transform: translateY(24px) scale(.98);} to {opacity:1; transform:none;} }
    .hero h1 { color:#fff; font-size: 3rem; font-weight: 800; margin:0; letter-spacing:.5px;
        text-shadow: 0 4px 24px rgba(0,0,0,.35); position:relative; z-index:1; }
    .hero p { color:#f3e8ff; font-size: 1.12rem; margin:.5rem 0 0; position:relative; z-index:1; max-width: 780px; }
    .hero .chips { margin-top: 1.1rem; position:relative; z-index:1; }
    .chip { display:inline-block; padding:.4rem .95rem; margin:.2rem .35rem .2rem 0; border-radius:999px;
        background: rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3); color:#fff;
        font-size:.82rem; font-weight:600; backdrop-filter: blur(6px); }

    /* ---- Glass cards ---- */
    .glass {
        background: linear-gradient(160deg, rgba(30,41,59,.72), rgba(17,24,39,.6));
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 20px; padding: 1.3rem 1.4rem; margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 12px 30px rgba(2,6,23,.45);
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
        animation: cardIn .7s ease both;
    }
    .glass:hover { transform: translateY(-6px);
        box-shadow: 0 22px 46px rgba(124,58,237,.30); border-color: rgba(167,139,250,.5); }
    @keyframes cardIn { from {opacity:0; transform: translateY(18px);} to {opacity:1; transform:none;} }
    .glass h3 { margin:.1rem 0 .5rem; font-size:1.15rem; }
    .glass .ic { font-size:1.7rem; }

    /* ---- Metric tiles ---- */
    .metric {
        border-radius: 16px; padding: .85rem .4rem; text-align:center;
        background: linear-gradient(160deg, rgba(2,6,23,.55), rgba(30,41,59,.5));
        border:1px solid rgba(148,163,184,.16);
        transition: transform .2s ease;
    }
    .metric:hover { transform: scale(1.05); }
    .metric .val { font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700;
        background: linear-gradient(90deg,#a5b4fc,#f0abfc); -webkit-background-clip:text;
        -webkit-text-fill-color:transparent; }
    .metric .lbl { font-size:.72rem; letter-spacing:.09em; text-transform:uppercase; color:#94a3b8; }

    /* ---- Badges / pills ---- */
    .badge { display:inline-block; padding:.32rem .85rem; border-radius:999px; font-weight:700;
        color:#fff; font-size:.85rem; box-shadow:0 6px 16px rgba(0,0,0,.3); }
    .rankpill { display:inline-flex; align-items:center; justify-content:center;
        width:34px; height:34px; border-radius:50%; font-weight:800; color:#fff; margin-right:10px;
        background: linear-gradient(135deg,#6366f1,#ec4899); box-shadow:0 6px 16px rgba(99,102,241,.5); }
    .winner {
        border: 1.5px solid rgba(52,211,153,.55) !important;
        box-shadow: 0 0 0 1px rgba(52,211,153,.25), 0 18px 44px rgba(16,185,129,.25) !important;
        animation: glow 2.2s ease-in-out infinite alternate;
    }
    @keyframes glow { from { box-shadow:0 0 0 1px rgba(52,211,153,.2), 0 12px 30px rgba(16,185,129,.18);}
                      to   { box-shadow:0 0 0 2px rgba(52,211,153,.4), 0 22px 52px rgba(16,185,129,.35);} }

    /* ---- Buttons ---- */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(120deg,#6366f1,#8b5cf6 50%,#ec4899);
        background-size: 200% 100%;
        color:#fff; border:none; border-radius:14px; padding:.65rem 1.5rem;
        font-weight:700; font-family:'Poppins'; letter-spacing:.3px;
        box-shadow:0 10px 26px rgba(124,58,237,.45);
        transition: transform .18s ease, background-position .5s ease, box-shadow .2s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-3px) scale(1.02); background-position: 100% 0;
        box-shadow:0 16px 34px rgba(236,72,153,.5); }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] { gap:8px; background:transparent; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30,41,59,.6); border:1px solid rgba(148,163,184,.18);
        border-radius:14px 14px 0 0; padding:10px 18px; color:#cbd5e1; font-weight:600;
        transition: all .2s ease; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg,#6366f1,#8b5cf6); color:#fff !important;
        border-color: transparent; box-shadow:0 8px 20px rgba(99,102,241,.4); }

    /* ---- Progress ---- */
    .stProgress > div > div > div { background: linear-gradient(90deg,#6366f1,#ec4899,#38bdf8); }

    /* ---- Uploader ---- */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(2,6,23,.4); border:1.5px dashed rgba(167,139,250,.5); border-radius:16px; }

    .section-title { font-family:'Space Grotesk'; font-size:1.5rem; font-weight:700; margin:.4rem 0 .2rem;
        background:linear-gradient(90deg,#c4b5fd,#f0abfc,#7dd3fc); -webkit-background-clip:text;
        -webkit-text-fill-color:transparent; }
    .muted { color:#94a3b8; font-size:.92rem; }
    img { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
PHOTOS_DIR = os.path.join(SAMPLES_DIR, "photos")
SKETCHES_DIR = os.path.join(SAMPLES_DIR, "sketches")


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_upload(uploaded) -> np.ndarray:
    return pil_to_bgr(Image.open(uploaded))


def load_sample_photos() -> list[dict]:
    photos = []
    if os.path.isdir(PHOTOS_DIR):
        for path in sorted(glob.glob(os.path.join(PHOTOS_DIR, "*.png"))):
            img = cv2.imread(path)
            if img is not None:
                photos.append({"name": os.path.basename(path), "image": img})
    return photos


def metric_tile(label: str, value: float) -> str:
    return (f"<div class='metric'><div class='val'>{value}%</div>"
            f"<div class='lbl'>{label}</div></div>")


# --------------------------------------------------------------------------- #
# HERO
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class='hero'>
        <h1>🎯 PhotoSketch</h1>
        <p>Match hand-drawn sketches to real photographs using computer vision and
        structural feature similarity — explainable, fast, and training-free.</p>
        <div class='chips'>
            <span class='chip'>🧠 Canny Edges</span>
            <span class='chip'>🔑 ORB Keypoints</span>
            <span class='chip'>📐 SSIM Fusion</span>
            <span class='chip'>⚡ No Training</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Live stat strip
_n = len(load_sample_photos())
s1, s2, s3, s4 = st.columns(4)
for col, (val, lbl) in zip(
    [s1, s2, s3, s4],
    [(f"{_n}", "Sample Photos"), ("3", "Fusion Signals"), ("256²", "Analysis Size"), ("0", "Training Needed")],
):
    col.markdown(metric_tile(lbl, val).replace("%", ""), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# MAIN TABS - all features on the main screen
# --------------------------------------------------------------------------- #
tab_match, tab_convert, tab_how = st.tabs(
    ["🔍  Sketch → Photo Matching", "✏️  Photo → Sketch Studio", "📖  How It Works"]
)


# ============================= TAB 1: MATCHING ============================= #
with tab_match:
    st.markdown("<div class='section-title'>Find the photo behind the sketch</div>",
                unsafe_allow_html=True)
    st.markdown("<p class='muted'>Provide a query sketch and a gallery of candidate photos. "
                "PhotoSketch ranks every photo by structural similarity.</p>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("<div class='glass'><span class='ic'>🖊️</span>"
                    "<h3>1 · Query Sketch</h3></div>", unsafe_allow_html=True)
        sketch_file = st.file_uploader("Upload a sketch", type=["png", "jpg", "jpeg"], key="sketch")
        use_sample_sketch = st.toggle("Use a bundled sample sketch instead", value=False)

        sketch_bgr = None
        if sketch_file is not None:
            # An uploaded sketch always takes priority
            sketch_bgr = read_upload(sketch_file)
        elif use_sample_sketch:
            sk_paths = sorted(glob.glob(os.path.join(SKETCHES_DIR, "*.png")))
            if sk_paths:
                names = [os.path.basename(p) for p in sk_paths]
                chosen = st.selectbox("Sample sketch", names)
                sketch_bgr = cv2.imread(os.path.join(SKETCHES_DIR, chosen))
            else:
                st.info("No sample sketches found. Run generate_samples.py.")

        if sketch_bgr is not None:
            st.image(bgr_to_rgb(sketch_bgr), caption="Query sketch", width=250)

    with right:
        st.markdown("<div class='glass'><span class='ic'>🖼️</span>"
                    "<h3>2 · Candidate Photos</h3></div>", unsafe_allow_html=True)
        source = st.radio("Photo source", ["Bundled samples", "Upload my own"], horizontal=True)

        photos = []
        if source == "Bundled samples":
            photos = load_sample_photos()
            st.markdown(f"<p class='muted'>{len(photos)} sample photos loaded.</p>",
                        unsafe_allow_html=True)
        else:
            photo_files = st.file_uploader("Upload photos", type=["png", "jpg", "jpeg"],
                                           accept_multiple_files=True, key="photos")
            for pf in photo_files or []:
                photos.append({"name": pf.name, "image": read_upload(pf)})
            st.markdown(f"<p class='muted'>{len(photos)} photos uploaded.</p>",
                        unsafe_allow_html=True)

        if photos:
            with st.expander(f"👁️ Preview gallery ({len(photos)} photos)", expanded=False):
                thumbs = st.columns(min(5, len(photos)))
                for i, p in enumerate(photos[:5]):
                    thumbs[i].image(bgr_to_rgb(p["image"]), width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🚀  Find Matches", width="stretch")

    if run:
        if sketch_bgr is None:
            st.error("Please provide a query sketch first.")
        elif not photos:
            st.error("Please provide at least one candidate photo.")
        else:
            prog = st.progress(0, text="Analysing structural features...")
            results = []
            for i, item in enumerate(photos):
                scores = engine.compare(sketch_bgr, item["image"])
                results.append({**item, **scores})
                prog.progress(int((i + 1) / len(photos) * 100),
                              text=f"Scoring {item['name']}...")
            results.sort(key=lambda r: r["score"], reverse=True)
            for rk, r in enumerate(results, 1):
                r["rank"] = rk
            prog.empty()

            best = results[0]
            label, color = engine.risk_free_label(best["score"])
            st.markdown(
                f"<div class='glass winner'><span class='rankpill'>1</span>"
                f"<b style='font-size:1.25rem'>🏆 Best match — {best['name']}</b> &nbsp;"
                f"<span class='badge' style='background:linear-gradient(120deg,{color},#8b5cf6)'>"
                f"{label} · {best['score']}%</span></div>", unsafe_allow_html=True)

            for r in results:
                label, color = engine.risk_free_label(r["score"])
                extra = "winner" if r["rank"] == 1 else ""
                with st.container():
                    st.markdown(f"<div class='glass {extra}'>", unsafe_allow_html=True)
                    ci, cinfo = st.columns([1, 3])
                    with ci:
                        st.image(bgr_to_rgb(r["image"]), width="stretch")
                    with cinfo:
                        st.markdown(
                            f"<span class='rankpill'>{r['rank']}</span>"
                            f"<b style='font-size:1.05rem'>{r['name']}</b> &nbsp;"
                            f"<span class='badge' style='background:linear-gradient(120deg,{color},#6366f1)'>"
                            f"{label}</span>", unsafe_allow_html=True)
                        st.progress(min(1.0, r["score"] / 100.0))
                        m1, m2, m3, m4 = st.columns(4)
                        m1.markdown(metric_tile("Overall", r["score"]), unsafe_allow_html=True)
                        m2.markdown(metric_tile("SSIM", r["ssim"]), unsafe_allow_html=True)
                        m3.markdown(metric_tile("ORB", r["orb"]), unsafe_allow_html=True)
                        m4.markdown(metric_tile("Edges", r["edge"]), unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)


# ============================ TAB 2: CONVERT =============================== #
with tab_convert:
    st.markdown("<div class='section-title'>Pencil Sketch Studio</div>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Turn any photograph into a pencil sketch and inspect the "
                "edge features the matcher relies on.</p>", unsafe_allow_html=True)

    photo_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"], key="convert")
    use_sample = st.toggle("Use a bundled sample photo instead", value=False, key="convsample")

    photo_bgr = None
    if photo_file is not None:
        # An uploaded photo always takes priority
        photo_bgr = read_upload(photo_file)
    elif use_sample:
        samples = load_sample_photos()
        if samples:
            names = [p["name"] for p in samples]
            chosen = st.selectbox("Sample photo", names, key="convpick")
            photo_bgr = next(p["image"] for p in samples if p["name"] == chosen)
        else:
            st.info("No sample photos found. Run generate_samples.py.")
    else:
        st.info("👆 Upload a photo above, or switch on the sample toggle to try a bundled image.")

    if photo_bgr is not None:
        blur = st.slider("Sketch softness", 5, 45, 21, step=2)
        sketch = engine.photo_to_sketch(photo_bgr, blur_ksize=blur)
        edges = engine.edge_map(engine.preprocess(photo_bgr))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='glass'><b>📷 Original</b></div>", unsafe_allow_html=True)
            st.image(bgr_to_rgb(photo_bgr), width="stretch")
        with c2:
            st.markdown("<div class='glass'><b>✏️ Pencil Sketch</b></div>", unsafe_allow_html=True)
            st.image(sketch, width="stretch", clamp=True)
        with c3:
            st.markdown("<div class='glass'><b>🧠 Edge Map</b></div>", unsafe_allow_html=True)
            st.image(edges, width="stretch", clamp=True)

        ok, buf = cv2.imencode(".png", sketch)
        if ok:
            st.download_button("⬇️  Download Sketch (PNG)", data=buf.tobytes(),
                               file_name="sketch.png", mime="image/png",
                               width="stretch")


# ============================ TAB 3: HOW ==================================== #
with tab_how:
    st.markdown("<div class='section-title'>How PhotoSketch works</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='glass'><span class='ic'>🧩</span><h3>The Modality Gap</h3>"
                    "<p class='muted'>A sketch and a photo of the same face look totally different "
                    "pixel-wise. We compare <b>structure</b>, not pixels.</p></div>",
                    unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass'><span class='ic'>🔬</span><h3>Feature Fusion</h3>"
                    "<p class='muted'>SSIM for global shape, ORB for local landmarks, edge IoU for "
                    "contours — fused into one explainable score.</p></div>",
                    unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='glass'><span class='ic'>🏆</span><h3>Ranked Results</h3>"
                    "<p class='muted'>Every candidate photo is scored and ranked, so the most "
                    "probable match rises to the top.</p></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='glass'>"
        "<h3>The Pipeline</h3>"
        "<p><b>1 · Preprocess</b> → resize to 256×256, grayscale, bilateral denoise, histogram equalise.<br>"
        "<b>2 · Bridge the gap</b> → convert each photo into the sketch domain "
        "(grayscale → invert → blur → colour-dodge).<br>"
        "<b>3 · Extract features</b> → Canny edge maps + ORB keypoints/descriptors.<br>"
        "<b>4 · Measure similarity</b> → SSIM (0.5) + ORB match ratio (0.3) + edge IoU (0.2).<br>"
        "<b>5 · Fuse & rank</b> → combine into one score and sort best-first.</p>"
        "</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='glass'><h3>🛠️ Tech Stack</h3>"
        "<p class='muted'>Python · OpenCV · NumPy · scikit-image · Pillow · Streamlit</p></div>",
        unsafe_allow_html=True)

    st.markdown(
        "<div class='glass'><h3>⚖️ Honest Limitations</h3>"
        "<p class='muted'>A classical, training-free system built for clarity. Best on clean, "
        "well-proportioned sketches, and intended as a <b>shortlisting aid</b> rather than a "
        "definitive identifier. A trained Siamese CNN would improve accuracy on rough forensic "
        "sketches — a natural future extension.</p></div>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;margin-top:1.4rem' class='muted'>"
    "Built with ❤️ using OpenCV & Streamlit — PhotoSketch</p>",
    unsafe_allow_html=True)
