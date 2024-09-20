import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.utils import CustomObjectScope
from PIL import Image

# Custom DepthwiseConv2D handler
def custom_depthwise_conv2d(*args, **kwargs):
    kwargs.pop('groups', None)  # Remove the 'groups' argument if present
    return DepthwiseConv2D(*args, **kwargs)

# Load the model once globally to avoid reloading and scope issues
@st.cache_resource
def load_mask_detection_model():
    with CustomObjectScope({'DepthwiseConv2D': custom_depthwise_conv2d}):
        return load_model("mask_detection_best.h5")

# Load the model only once
model = load_mask_detection_model()

# Define a function to detect masks in the image
def mask_detection_image(image, model):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        face_roi_resized = cv2.resize(face_roi, (224, 224))
        face_roi_preprocessed = np.expand_dims(face_roi_resized / 255.0, axis=0)

        try:
            prediction = model.predict(face_roi_preprocessed)
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            return image

        label = "Mask" if prediction[0][0] > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return image

# Streamlit app
def main():
    st.title("Face Mask Detection Web App")

    st.write("Upload an image or use your webcam to detect face masks.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image = np.array(image)
            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            result_image = mask_detection_image(image, model)
            st.image(result_image, caption="Processed Image", use_column_width=True)

        except Exception as e:
            st.error(f"Error processing image: {e}")

    # Initialize session state for controlling the webcam
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False

    # Webcam functionality
    if st.session_state.webcam_active:
        if st.button("Stop Webcam", key="stop_button"):
            st.session_state.webcam_active = False  # Stop webcam

        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture video")
                break

            frame = mask_detection_image(frame, model)
            stframe.image(frame, channels="BGR")

        cap.release()
        stframe.empty()  # Clear the webcam feed display after stopping
    else:
        if st.button("Start Webcam", key="start_button"):
            st.session_state.webcam_active = True  # Start webcam

if __name__ == "__main__":
    main()
