import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('D:\capstone project\Flask\Model.h5')

# Preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    return img_array

# Classes (put dataset classes)
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Perform image classification
def classify_image(image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    class_indices = np.argmax(prediction, axis=1)
    class_labels = [classes[i] for i in class_indices]
    probabilities = np.max(prediction, axis=1)

    results = []
    for label, probability in zip(class_labels, probabilities):
        results.append({'label': label, 'probability': float(probability)})

    return results

# Flask route for index page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Flask route for image classification
@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image_file = request.files['image']
    image = Image.open(image_file)

    # Convert the image to base64 string
    image_data = io.BytesIO()
    image.save(image_data, format='JPEG')
    encoded_image = base64.b64encode(image_data.getvalue()).decode('utf-8')

    results = classify_image(image)

    return render_template('predict.html', image=encoded_image, results=results)
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=9877)
