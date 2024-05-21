from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Load the processor and model
processor = AutoImageProcessor.from_pretrained("CarViT")
model = AutoModelForImageClassification.from_pretrained("CarViT")

# Load the image
image = Image.open("CarViT/images/Acura.jpg")

# Preprocess image
inputs = processor(images=image, return_tensors="pt")

# Make predictions
outputs = model(**inputs)

# Get probabilities
predictions = outputs.logits.squeeze().softmax(dim=0)

# Get predicted class index and probability
predicted_class = predictions.argmax().item()
predicted_prob = predictions[predicted_class].item()

print(f"Predicted class: {predicted_class}, Probability: {predicted_prob}")
