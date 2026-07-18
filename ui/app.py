import sys
import time
from pathlib import Path
import torch
import streamlit as st
from PIL import Image
from torchvision.transforms import v2

# allow absolute imports from project root
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# pyrefly: ignore [missing-import]
from src.data.dataset import MathDataset
# pyrefly: ignore [missing-import]
from src.data.tokenizer import HybridTokenizer
# pyrefly: ignore [missing-import]
from src.models.encoder import VisionEncoder
# pyrefly: ignore [missing-import]
from src.models.decoder import LatexDecoder
# pyrefly: ignore [missing-import]
from src.inference import generate

import base64

# Page config — must be first Streamlit call
try:
    icon_image = Image.open(project_root / "ui" / "img" / "mini_logo.png")
    # Crop out transparent padding so the icon fills the browser tab
    bbox = icon_image.getbbox()
    if bbox:
        icon_image = icon_image.crop(bbox)
except Exception:
    icon_image = "📷"

st.set_page_config(
    page_title="ScanTex",
    page_icon=icon_image,
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* hide default streamlit header/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* page background */
.stApp {
    background-color: #f0f8ff;
}

/* main card */
.main-card {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
}

/* big brand title */
.brand-title {
    font-size: 3.2rem;
    font-weight: 800;
    color: #0d2137;
    letter-spacing: -1px;
    margin-bottom: 0.1rem;
}

.brand-subtitle {
    font-size: 1rem;
    color: #6b7a8d;
    margin-bottom: 2rem;
}

/* column headers */
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0d2137;
    margin-bottom: 0.8rem;
}

/* code block override */
.stCode {
    background: #1e2533 !important;
    border-radius: 8px;
}

/* metric cards */
.metric-card {
    background: #f0f3f7;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    display: inline-block;
    min-width: 120px;
}

.metric-label {
    font-size: 0.78rem;
    color: #6b7a8d;
    margin-bottom: 2px;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0d2137;
}

/* Run OCR button */
div[data-testid="stButton"] button {
    background-color: #1b2d4f;
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    width: 100%;
    font-size: 1rem;
    transition: background 0.2s;
}

div[data-testid="stButton"] button:hover {
    background-color: #254068;
}

/* Compiles badge */
.badge-compiles {
    display: inline-block;
    background: #d1fae5;
    color: #065f46;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 999px;
    padding: 2px 10px;
    margin-left: 8px;
    vertical-align: middle;
}

/* uploaded file info */
.file-info-row {
    background: #f0f3f7;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-top: 0.6rem;
    font-size: 0.88rem;
    color: #374151;
}

/* compiled preview box */
.preview-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    min-height: 100px;
}

/* divider */
.section-divider {
    border: none;
    border-top: 1px solid #e8ecf0;
    margin: 1.5rem 0;
}

/* Upload sections in light blue and white */
div[data-testid="stFileUploaderDropzone"] {
    background-color: #f0f9ff;
    border: 2px dashed #38bdf8;
    border-radius: 8px;
}
div[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    if torch.cuda.is_available():      device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else:                               device = torch.device("cpu")
    
    dataset = MathDataset()
    tokenizer = HybridTokenizer(vocab_size=4000)
    tokenizer.train(dataset.latex_strings)
    
    checkpoint_path = project_root / "checkpoint.pt"
    if not checkpoint_path.exists():
        st.error("checkpoint.pt not found. Run training first.")
        st.stop()
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    encoder = VisionEncoder().to(device)
    decoder = LatexDecoder(vocab_size=checkpoint['vocab_size']).to(device)
    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, tokenizer, device


encoder, decoder, tokenizer, device = load_models()

# Brand header
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    logo_b64 = get_base64_image(project_root / "ui" / "img" / "Full_logo.png")
    st.markdown(
        f'''
        <div style="background-color: #0d2137; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; display: flex; align-items: center; justify-content: flex-start;">
            <img src="data:image/png;base64,{logo_b64}" style="max-height: 80px;" alt="ScanTex Logo">
        </div>
        ''',
        unsafe_allow_html=True
    )
except Exception:
    st.markdown('<div class="brand-title">ScanT<span style="font-family:serif; font-style:italic;">E</span>X</div>', unsafe_allow_html=True)

st.markdown('<div class="brand-subtitle">Upload a scanned equation image and get compilable LaTeX back.</div>', unsafe_allow_html=True)

# Two-column layout
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="section-header">Upload image</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="PNG, JPG supported. Max 200 MB."
    )
    
    # Show file info row when file is uploaded
    if uploaded_file is not None:
        size_kb = uploaded_file.size // 1024
        image = Image.open(uploaded_file).convert("RGB")
        w, h = image.size
        st.markdown(
            f'<div class="file-info-row">🖼️ <strong>{uploaded_file.name}</strong>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;Resolution: {w}×{h} — {size_kb} KB</div>',
            unsafe_allow_html=True
        )
        st.image(image, width='stretch')
    
    run_button = st.button("▶  Run OCR", disabled=(uploaded_file is None))


# Right column — results (always visible, populated after running)
with right_col:
    st.markdown('<div class="section-header">Generated code</div>', unsafe_allow_html=True)
    
    code_placeholder = st.empty()
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    st.markdown(
        '<div class="section-header">Compiled preview '
        '<span class="badge-compiles" id="badge-compiles" style="display:none">Compiles</span>'
        '</div>',
        unsafe_allow_html=True
    )
    
    preview_placeholder = st.empty()

# Default placeholder state
if "result_latex" not in st.session_state:
    with code_placeholder:
        st.code("# Output will appear here after running OCR", language="latex")
    with preview_placeholder:
        st.markdown('<div class="preview-box" style="color:#9ca3af;">Preview will appear here.</div>', unsafe_allow_html=True)

# Metric placeholders below columns
m1, spacer = st.columns([1, 5])
latency_placeholder = m1.empty()


# --- Run inference on button click ---
if run_button and uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("Running OCR..."):
        img_tensor = v2.functional.pil_to_tensor(image)
        img_tensor = v2.functional.to_dtype(img_tensor, torch.float32, scale=True)
        
        t0 = time.perf_counter()
        predicted_latex = generate(img_tensor, encoder, decoder, tokenizer, device)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
    
    st.session_state["result_latex"] = predicted_latex
    st.session_state["elapsed_ms"] = elapsed_ms

# Render results from session state
if "result_latex" in st.session_state:
    latex_str = st.session_state["result_latex"]
    elapsed_ms = st.session_state["elapsed_ms"]
    
    with code_placeholder:
        st.code(latex_str, language="latex")
    
    # Check if LaTeX compiles via st.latex
    with preview_placeholder:
        st.markdown('<div class="badge-compiles">Compiles</div>', unsafe_allow_html=True)
        st.latex(latex_str)
    
    with latency_placeholder:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Latency</div>'
            f'<div class="metric-value">{elapsed_ms} ms</div></div>',
            unsafe_allow_html=True
        )
