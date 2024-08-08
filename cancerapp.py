from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
model=tf.keras.models.load_model('skin_lesion_model.keras')
class_names =['Acitinic Keratosis',
 'Basal Cell Carcinoma',
 'Dermatofibroma',
 'Melanoma',
 'Nevus',
 'Pigmented Benign Keratosis',
 'Seborrheic Keratosis',
 'Squamous Cell Carcinoma',
 'Vascular Lesion']


# preprocessing of the image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  
    return img_array

def predict_image(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions, axis=1)
    return predicted_class, confidence


@app.route('/cancer_predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    base_path = 'sample images'
    filename = 'melanoma(1).jpg'
    img_path = os.path.join(base_path, filename)
    file.save(img_path)
    predicted_class, confidence = predict_image(model, img_path)

    # Interpret the results
    response = {
        'Predicted Class': class_names[predicted_class[0]],
        'Confidence': float(confidence[0])
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)