# Face Mask Detection Web App

This web application detects whether a person in an image or live video feed is wearing a face mask or not. It uses a pre-trained deep learning model and provides real-time detection via image uploads or webcam input.

## About

This project leverages machine learning techniques and computer vision to build a Face Mask Detection system. The app is built using **Streamlit** for easy deployment and interface, **OpenCV** for image processing, and **TensorFlow** to run the pre-trained model for mask detection. It also integrates **streamlit-webrtc** to capture live video from the user's browser for real-time mask detection.

## Features

- Detects masks in uploaded images.
- Real-time mask detection via webcam (using the browser).
- Works on both desktop and mobile devices.
- Simple and intuitive interface.

## How It Works

1. **Image Upload**: Users can upload an image file in `.jpg`, `.jpeg`, or `.png` format.
2. **Webcam Integration**: Users can activate their webcam to detect masks in real-time video streams directly from their browser.
3. **Mask Detection**: The application detects faces using OpenCV's Haar Cascade classifier and runs each face through a pre-trained model to check for a mask.

## Tech Stack

- **Python**
- **Streamlit** for building the web application.
- **OpenCV** for image processing and face detection.
- **TensorFlow** for loading and running the pre-trained mask detection model.
- **PIL (Pillow)** for image file handling.
- **streamlit-webrtc** for real-time webcam streaming through the browser.

![1718186275331](https://github.com/user-attachments/assets/a1f1a20d-fa42-46a8-a186-dd0768548d37)
![1718186274861](https://github.com/user-attachments/assets/a7d5672a-3e4c-4602-b3a1-e554880cabd0)
