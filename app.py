import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="COVID-19 Chest X-Ray Detection",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_covid_model():
    return tf.keras.models.load_model("covid_model.keras")

model = load_covid_model()

# -----------------------------
# Title
# -----------------------------
st.title("🩺 COVID-19 Detection from Chest X-Ray")
st.markdown(
    """
    Upload a Chest X-Ray image and the model will predict whether it is:

    - 🦠 **COVID**
    - ✅ **NORMAL**
    """
)

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded X-Ray",
        use_container_width=True
    )

    # Preprocessing
    img = image.resize((299, 299))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    probability = float(prediction[0][0])

    st.subheader("Prediction Result")

    if probability > 0.5:
        st.error("🦠 COVID Detected")
        st.write(f"Confidence: {probability:.2%}")
    else:
        st.success("✅ NORMAL")
        st.write(f"Confidence: {(1 - probability):.2%}")

    st.progress(probability if probability > 0.5 else 1 - probability)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    **Developed by:** Richeek Pandey

    🔗 LinkedIn: www.linkedin.com/in/richeek-pandey-2604a3335

    💻 GitHub: https://github.com/richeekpandey07
    """
)