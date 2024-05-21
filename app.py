from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import streamlit as st  

processor = AutoImageProcessor.from_pretrained("CarViT")
model = AutoModelForImageClassification.from_pretrained("CarViT")
def classify_image(image):
    """Classifies an image using the CarViT model.

    Args:
        image: A PIL Image object or NumPy array representing the image.

    Returns:
        A tuple containing the predicted class and its probability.
    """

    inputs = processor(image=image, return_tensors="pt")  # Preprocess image
    outputs = model(**inputs)  # Make predictions
    predictions = outputs.logits.squeeze().softmax(dim=0)  # Get probabilities
    predicted_class = predictions.argmax().item()  # Get predicted class index
    predicted_prob = predictions[predicted_class].item()  # Get probability for predicted class

    return predicted_class, predicted_prob
uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Read uploaded image
    st.image(image, caption="Uploaded Image")

    # Classify image only if a valid image is uploaded
    if image.mode == 'RGB':
        predicted_class, predicted_prob = classify_image(image)
        class_names = {  # Assuming you have class names (modify as needed)
            0: "Car",
            1: "Truck",
            2: "Motorcycle",
            # Add more classes as needed
        }

        predicted_class_name = class_names[predicted_class]
        st.success(f"Predicted Class: {predicted_class_name} (Probability: {predicted_prob:.4f})")
    else:
        st.warning("Please upload an RGB image (JPG, JPEG, or PNG).")

else:
    st.info("Upload an image to start classification.")