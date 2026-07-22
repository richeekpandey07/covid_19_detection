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
#     """
# )

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="COVID-19 X-Ray Detection",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #ffffff;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
}

.footer {
    text-align: center;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")
    return model

model = load_model()

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.title("🩺 COVID Detector")

    st.info("""
    Upload a Chest X-Ray image.

    AI will classify:
    - 🦠 COVID
    - ✅ NORMAL
    """)

    st.markdown("---")

    st.subheader("Project Info")

    st.write("""
    Deep Learning based Chest X-Ray
    Classification System using
    TensorFlow & Streamlit.
    """)

# -------------------------------
# Main Title
# -------------------------------
st.title("🩺 COVID-19 Detection from Chest X-Ray")
st.write(
    "Upload a Chest X-Ray image and get an AI-powered prediction."
)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Session State History
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded X-Ray",
            use_container_width=True
        )

    with st.spinner("Analyzing X-Ray..."):

        # Resize according to model input
        img = image.resize((299, 299))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        probability = float(prediction[0][0])

    with col2:

        st.subheader("Prediction Result")

        if probability > 0.5:
            result = "COVID"
            confidence = probability

            st.error(
                f"🦠 COVID DETECTED\n\nConfidence: {confidence:.2%}"
            )

        else:
            result = "NORMAL"
            confidence = 1 - probability

            st.success(
                f"✅ NORMAL\n\nConfidence: {confidence:.2%}"
            )

        st.progress(float(confidence))

        # Confidence Chart
        chart_data = pd.DataFrame({
            "Class": ["Prediction"],
            "Confidence": [confidence]
        })

        st.subheader("Confidence Score")
        st.bar_chart(
            chart_data.set_index("Class")
        )

        # Result Table
        result_df = pd.DataFrame({
            "Prediction": [result],
            "Confidence": [f"{confidence:.2%}"]
        })

        st.table(result_df)

    # -------------------------------
    # Save History
    # -------------------------------
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Prediction": result,
        "Confidence": f"{confidence:.2%}"
    })

    # -------------------------------
    # Download Report
    # -------------------------------
    report = f"""
COVID-19 Detection Report
=========================

Prediction : {result}
Confidence : {confidence:.2%}

Generated On :
{datetime.now()}

Developer :
Richeek Pandey

Disclaimer:
This AI prediction is for educational
purposes only and should not be used
as a medical diagnosis.
"""

    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name="covid_report.txt",
        mime="text/plain"
    )

# -------------------------------
# Prediction History
# -------------------------------
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

# -------------------------------
# Disclaimer
# -------------------------------
st.markdown("---")

st.warning("""
⚠️ Medical Disclaimer

This application is intended for educational and research purposes only.

The prediction generated by the AI model should NOT be considered a final medical diagnosis.

Please consult a qualified healthcare professional for any medical advice.
""")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")

st.markdown("""
<div class="footer">

### Developed By Richeek Pandey

🔗 LinkedIn:
https://www.linkedin.com/in/richeek-pandey-9954783a9

💻 GitHub:
https://github.com/richeekpandey07

</div>
""", unsafe_allow_html=True)
