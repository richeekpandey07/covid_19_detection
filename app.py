import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon="🩺",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
}

h1, h2, h3 {
    color: white !important;
}

p, label, div {
    color: blue;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: red;
}

.footer {
    text-align: center;
    color: white;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# MODEL LOAD
# =====================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# =====================================
# SESSION HISTORY
# =====================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================
# SIDEBAR
# =====================================
with st.sidebar:
    st.title("🩺 COVID Detector")

    st.success("Model Loaded")

    st.markdown("---")

    st.markdown("""
    ### Features

    ✅ AI Prediction

    ✅ Confidence Score

    ✅ Download Report

    ✅ Prediction History
    """)

# =====================================
# TITLE
# =====================================
st.title("🩺 COVID-19 Chest X-Ray Detection")

st.write(
    "Upload a chest X-ray image and get an AI-powered prediction."
)

# =====================================
# FILE UPLOADER
# =====================================
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDICTION
# =====================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with st.spinner("Analyzing..."):

        img = image.resize((299, 299))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        probability = float(prediction[0][0])

        if probability > 0.5:
            result = "COVID"
            confidence = probability
        else:
            result = "NORMAL"
            confidence = 1 - probability

    with col2:

        st.subheader("Prediction Result")

        if result == "COVID":
            st.markdown(
                f"""
                <div class='result-box'
                style='background:#dc2626'>
                🦠 COVID DETECTED
                <br><br>
                {confidence:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class='result-box'
                style='background:#059669'>
                ✅ NORMAL
                <br><br>
                {confidence:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        m1, m2, m3 = st.columns(3)

        m1.metric("Prediction", result)
        m2.metric("Confidence", f"{confidence:.2%}")
        m3.metric("Model", "CNN")

    # Chart
    st.subheader("Confidence Score")

    chart_df = pd.DataFrame({
        "Confidence": [confidence]
    })

    st.bar_chart(chart_df)

    # Save history
    st.session_state.history.append({
        "Time": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "Prediction": result,
        "Confidence": f"{confidence:.2%}"
    })

    # Download Report
    report = f"""
COVID-19 DETECTION REPORT

Prediction: {result}
Confidence: {confidence:.2%}

Generated On:
{datetime.now()}
"""

    st.download_button(
        "📄 Download Report",
        report,
        file_name="covid_report.txt",
        mime="text/plain"
    )

# =====================================
# HISTORY
# =====================================
if len(st.session_state.history) > 0:

    st.subheader("📜 Prediction History")

    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True
    )

# =====================================
# DISCLAIMER
# =====================================
st.warning(
    "⚠️ This application is for educational purposes only. "
    "Please consult a healthcare professional for diagnosis."
)

# =====================================
# FOOTER
# =====================================
st.markdown(
    """
    <div class='footer'>
    <hr>
    <h3>👨‍💻 Developed By Richeek Pandey</h3>

    LinkedIn:
    https://www.linkedin.com/in/richeek-pandey-2604a3335

    <br><br>

    GitHub:
    https://github.com/richeekpandey07
    </div>
    """,
    unsafe_allow_html=True
)

