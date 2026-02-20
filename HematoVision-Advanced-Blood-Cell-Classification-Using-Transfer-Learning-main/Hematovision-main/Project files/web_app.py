import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'blood_cell_classifier_mobilenetv2.h5'
# Check if model exists before loading
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please run 'python app.py' first to train and save the model.")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Alphabetical order matches flow_from_directory)
CLASS_LABELS = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

# Configure upload folder
UPLOAD_FOLDER = 'static/images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match training

    # Make prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return CLASS_LABELS[class_idx], float(confidence)

@app.route('/')
@app.route('/home.html')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        prediction, confidence = predict_image(file_path)
        
        return render_template('result.html',image_file=filename,prediction=prediction,confidence=f"{confidence*100:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)