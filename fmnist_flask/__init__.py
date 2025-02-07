import tensorflow as tf
import numpy as np
import cv2
import flask
from flask import Flask, request, jsonify

# Load the trained model
model = tf.keras.models.load_model("fmnist.h5")

# Class labels
classes = {
    0: 'T-shirt/top',
    1: 'trouser',
    2: 'pullover',
    3: 'dress',
    4: 'coat',
    5: 'sandal',
    6: 'shirt',
    7: 'sneaker',
    8: 'bag',
    9: 'ankle boot'
}

app = Flask(__name__)

def preprocess_image(image_path):
    """Loads and preprocesses an image for inference."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if image is None:
        return None
    image = cv2.resize(image, (28, 28))  # Resize to 28x28
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image classification requests."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image_path = "/tmp/uploaded_image.png"  # Temporary file
    file.save(image_path)

    image = preprocess_image(image_path)
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    return jsonify({"predicted_class": classes[predicted_class]})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8083)
