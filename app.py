import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    background-color: #F5F5F5;
}

h1, h2, h3 {
    color: #2F4157;
}

p, li {
    color: #577C8E;
    font-size: 15px;
}

.section-card {
    background-color: #FFFFFF;
    padding: 24px;
    border-radius: 14px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    margin-bottom: 24px;
}

.guidelines li {
    margin-bottom: 8px;
}

.result-box {
    background-color: #E0E5E8;
    padding: 20px;
    border-radius: 12px;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# CONFIG
# ======================
IMG_SIZE = 224
THRESHOLD = 0.69   # ⚠️ GANTI kalau kamu punya best_thresh dari training

st.set_page_config(
    page_title="Skin Cancer Detector",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<div class="section-card">
    <h1>🩺 Skin Cancer Detection</h1>
    <p>
        This application uses a deep learning model to analyze skin lesion images 
        and estimate whether the lesion is more likely to be 
        <b>Benign (non-cancerous)</b> or <b>Malignant (potentially cancerous)</b>.
    </p>
</div>
""", unsafe_allow_html=True)


# ======================
# LOAD MODEL (CACHED)
# ======================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "resnet_95_fold_3.keras",
        compile=False
    )

model = load_model()

# ======================
# PREPROCESSING (MATCH TRAINING)
# ======================
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32)

    img = preprocess_input(img)  # 🔥 WAJIB untuk ResNet50
    return np.expand_dims(img, axis=0)

st.markdown("""
<div class="section-card">
    <h3>📸 Image Upload Guidelines</h3>
    <p>
        For best prediction accuracy, please follow these recommendations when taking the photo:
    </p>
    <ul class="guidelines">
        <li>Ensure the skin lesion is clearly visible and in focus</li>
        <li>Use sufficient natural or white lighting (avoid shadows)</li>
        <li>Keep the camera at a moderate distance (not too close or too far)</li>
        <li>Avoid using filters or beauty effects</li>
        <li>Clean the skin area so the lesion is unobstructed</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ======================
# UI
# ======================
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    img_array = preprocess_image(image)
    preds = model.predict(img_array, verbose=0)[0]

    p_benign = float(preds[0])
    p_malignant = float(preds[1])

    st.markdown("""
    <div class="section-card">
        <h3>🔍 Prediction Result</h3>
        <p>
            <b>Confidence score</b> represents how confident the model is in its prediction,
            based on patterns learned from medical image data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-box">
        <p><b>Benign probability:</b> {p_benign:.3f}</p>
        <p><b>Malignant probability:</b> {p_malignant:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

    if p_malignant >= THRESHOLD:
        st.error("""
        ⚠️ **Malignant (Potentially Cancerous)**

        The model indicates a higher probability that this lesion may be malignant.
        This does **not** confirm a medical diagnosis.

        **Recommendation:**
        Please consult a qualified healthcare professional or dermatologist
        for further examination and diagnosis as soon as possible.
        """)
    else:
        st.success("""
        ✅ **Benign (Non-cancerous)**

        The model suggests that the lesion is more likely benign.
        While this is generally not dangerous, it is still recommended to
        monitor any changes in size, color, or shape over time.

        **Recommendation:**
        Seek medical advice if you notice unusual changes or symptoms.
        """)

st.markdown("""
<p style="font-size:12px; color:#577C8E; text-align:center; margin-top:40px;">
⚠️ This tool is intended for educational and research purposes only.
It does not replace professional medical diagnosis or advice.
</p>
""", unsafe_allow_html=True)
