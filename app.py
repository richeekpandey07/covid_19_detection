# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # -----------------------------
# # Page Config
# # -----------------------------
# st.set_page_config(
#     page_title="COVID-19 Chest X-Ray Detection",
#     page_icon="🩺",
#     layout="centered"
# )

# # -----------------------------
# # Load Model
# # -----------------------------
# @st.cache_resource
# def load_covid_model():
#     return tf.keras.models.load_model("model.keras")

# model = load_covid_model()

# # -----------------------------
# # Title
# # -----------------------------
# st.title("🩺 COVID-19 Detection from Chest X-Ray")
# st.markdown(
#     """
#     Upload a Chest X-Ray image and the model will predict whether it is:

#     - 🦠 **COVID**
#     - ✅ **NORMAL**
#     """
# )

# # -----------------------------
# # Upload Image
# # -----------------------------
# uploaded_file = st.file_uploader(
#     "Upload Chest X-Ray Image",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:

#     image = Image.open(uploaded_file).convert("RGB")

#     st.image(
#         image,
#         caption="Uploaded X-Ray",
#         use_container_width=True
#     )

#     # Preprocessing
#     img = image.resize((299, 299))

#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Prediction
#     prediction = model.predict(img_array)

#     probability = float(prediction[0][0])

#     st.subheader("Prediction Result")

#     if probability > 0.5:
#         st.error("🦠 COVID Detected")
#         st.write(f"Confidence: {probability:.2%}")
#     else:
#         st.success("✅ NORMAL")
#         st.write(f"Confidence: {(1 - probability):.2%}")

#     st.progress(probability if probability > 0.5 else 1 - probability)

# # -----------------------------
# # Footer
# # -----------------------------
# st.markdown("---")
# st.markdown(
#     """
#     **Developed by:** Richeek Pandey

#     🔗 LinkedIn: www.linkedin.com/in/richeek-pandey-2604a3335

#     💻 GitHub: https://github.com/richeekpandey07
st.markdown("""
<style>

.stApp{
    background-color:#0f172a;
}

/* Main Title */
.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:white;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
}

/* Glass Card */
.glass{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius:20px;
    padding:25px;
    border:1px solid rgba(255,255,255,0.1);
    box-shadow:0 8px 32px rgba(0,0,0,0.3);
}

.result-card{
    border-radius:20px;
    padding:20px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
    color:white;
}

section[data-testid="stSidebar"]{
    background:#111827;
}

.stButton>button{
    width:100%;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#111827;
}

/* Buttons */
.stButton>button{
    width:100%;
    border-radius:12px;
    background:#06b6d4;
    color:white;
    font-weight:bold;
}

/* Upload Box */
[data-testid="stFileUploader"]{
    border:2px dashed #06b6d4;
    border-radius:15px;
    padding:10px;
}

</style>
""", unsafe_allow_html=True)

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import requests
from streamlit_lottie import st_lottie

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="COVID-19 AI Detector",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

.stApp{
    background: linear-gradient(
    135deg,
    #0f172a,
    #1e293b,
    #0f172a);
}

/* Title */
.main-title{
    text-align:center;
    color:white;
    font-size:48px;
    font-weight:700;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
}

/* Glass Effect */
.glass{
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(12px);
    padding:25px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.1);
    box-shadow:0px 8px 32px rgba(0,0,0,0.3);
}

/* Result Card */
.result-card{
    padding:25px;
    border-radius:20px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:white;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#111827;
}

/* Upload Box */
[data-testid="stFileUploader"]{
    border:2px dashed #06b6d4;
    border-radius:15px;
    padding:10px;
}

/* Footer */
.footer{
    text-align:center;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOTTIE FUNCTION
# =====================================================
def load_lottie(url):
    try:
        return requests.get(url).json()
    except:
        return None

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
        width=120
    )

    st.title("🩺 AI Medical Assistant")

    st.success("Model Loaded Successfully")

    st.markdown("---")

    st.markdown("""
    ### Features

    ✅ Deep Learning Model

    ✅ Real-Time Detection

    ✅ Confidence Score

    ✅ Prediction History

    ✅ Download Report

    ✅ Streamlit Dashboard
    """)

    st.markdown("---")

    st.info("""
    Upload a Chest X-Ray image
    to detect COVID-19.
    """)

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div class="glass">

<h1 class="main-title">
🩺 COVID-19 X-Ray AI Detector
</h1>

<p class="subtitle">
Deep Learning Powered Chest X-Ray Classification System
</p>

</div>
""", unsafe_allow_html=True)

# =====================================================
# LOTTIE ANIMATION
# =====================================================
lottie_url = "https://assets4.lottiefiles.com/packages/lf20_0yfsb3a1.json"

lottie_json = load_lottie(lottie_url)

if lottie_json:
    st_lottie(
        lottie_json,
        height=250,
        key="medical"
    )

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# PREDICTION
# =====================================================
if uploaded_file is not None:

    col1, col2 = st.columns([1,1])

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(
            image,
            caption="Uploaded X-Ray",
            use_container_width=True
        )

    with st.spinner("🔍 Analyzing X-Ray..."):

        img = image.resize((299,299))

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

        st.subheader("📊 Prediction Result")

        if result == "COVID":

            st.markdown(f"""
            <div class="result-card"
            style="background:linear-gradient(135deg,#ef4444,#b91c1c)">
            🦠 COVID DETECTED
            <br><br>
            {confidence:.2%}
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-card"
            style="background:linear-gradient(135deg,#10b981,#047857)">
            ✅ NORMAL
            <br><br>
            {confidence:.2%}
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        colA, colB, colC = st.columns(3)

        colA.metric(
            "Prediction",
            result
        )

        colB.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        colC.metric(
            "Model",
            "CNN"
        )

        st.progress(float(confidence))

    # =====================================================
    # CHART
    # =====================================================
    st.subheader("📈 Confidence Score")

    chart_df = pd.DataFrame({
        "Class":["Prediction"],
        "Confidence":[confidence]
    })

    st.bar_chart(
        chart_df.set_index("Class")
    )

    # =====================================================
    # RESULT TABLE
    # =====================================================
    st.subheader("📋 Detailed Report")

    report_df = pd.DataFrame({
        "Prediction":[result],
        "Confidence":[f"{confidence:.2%}"],
        "Timestamp":[datetime.now()]
    })

    st.dataframe(
        report_df,
        use_container_width=True
    )

    # =====================================================
    # SAVE HISTORY
    # =====================================================
    st.session_state.history.append({
        "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Prediction": result,
        "Confidence": f"{confidence:.2%}"
    })

    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================
    report_text = f"""
COVID-19 DETECTION REPORT
========================

Prediction : {result}

Confidence : {confidence:.2%}

Generated On :
{datetime.now()}

Developer :
Richeek Pandey

DISCLAIMER:
This tool is intended for educational
and research purposes only.

Please consult a qualified doctor
for any medical diagnosis.
"""

    st.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name="covid_report.txt",
        mime="text/plain"
    )

# =====================================================
# HISTORY
# =====================================================
if len(st.session_state.history) > 0:

    st.markdown("---")

    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

# =====================================================
# DISCLAIMER
# =====================================================
st.markdown("---")

st.warning("""
⚠️ Medical Disclaimer

This application is intended for educational and research purposes only.

The prediction generated by the AI model should NOT be considered a final medical diagnosis.

Always consult a qualified healthcare professional.
""")

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">

<hr>

<h3>👨‍💻 Developed By Richeek Pandey</h3>

🔗 LinkedIn<br>
https://www.linkedin.com/in/richeek-pandey-2604a3335

<br><br>

💻 GitHub<br>
https://github.com/richeekpandey07

</div>
""", unsafe_allow_html=True)
