import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def model_prediction(test_image):
    model = tf.keras.models.load_model("plant-disease-detection-model.keras")
    image = Image.open(test_image).convert('RGB')
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0  # Normalize the image
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar Styling
st.sidebar.title("🌱 Plant Disease Detection")
st.sidebar.markdown("### Sustainable Agriculture")

app_mode = st.sidebar.radio("Select Page", ["🏠 Home", "🔍 Disease Recognition"])

# Display Banner Image
st.image("AI-agriculture.jpg", use_container_width=True)

# Main Page
if app_mode == "🏠 Home":
    st.markdown("""
        <h1 style='text-align: center; color: green;'>🌿 Plant Disease Detection System</h1>
        <p style='text-align: center;'>Empowering sustainable agriculture with AI-driven disease recognition.</p>
        <hr>
        <h3>📌 How It Works?</h3>
        <ul>
            <li>Upload an image or capture one using your camera.</li>
            <li>Our deep learning model analyzes the image.</li>
            <li>Get instant predictions on the detected disease.</li>
        </ul>
    """, unsafe_allow_html=True)

elif app_mode == "🔍 Disease Recognition":
    st.header("🌱 Plant Disease Recognition")
    st.markdown("Upload an image or take a picture to get a disease prediction.")

    method = st.sidebar.selectbox("Choose Input Method", ["Upload Image", "Use Camera"])
    test_image = None

    if method == "Upload Image":
        test_image = st.file_uploader("📸 Choose an Image", type=["jpg", "jpeg", "png"])
    elif method == "Use Camera":
        test_image = st.camera_input("📷 Take a Picture")

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
        predict_button = st.button("🔎 Predict Disease")

        if predict_button:
            with st.spinner("Analyzing... 🧐"):
                result_index = model_prediction(test_image)
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                              'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                              'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                              'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_spot)',
                              'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                              'Peach___healthy', 'Pepper,bell___Bacterial_spot', 'Pepper,bell___healthy',
                              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                              'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                              'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_curl_Virus', 'Tomato___Tomato_mosaic_virus',
                              'Tomato___healthy']
                
                plant_name, disease_name = class_name[result_index].split("___") if "___" in class_name[result_index] else (class_name[result_index], "Healthy")
                
                st.success(f"🌿 Plant Name: {plant_name}\n🦠 Disease Name: {disease_name}")
                
                st.balloons()
