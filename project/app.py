from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model (replace with your actual model)
MODEL_PATH = 'models/insulator_defect_model.h5'
model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0

def detect_defects(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    defects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:
            x, y, w, h = cv2.boundingRect(contour)
            defects.append({'x': x, 'y': y, 'severity': 'high' if area > 1000 else 'medium'})
    return defects

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not all(f'{view}' in request.files for view in ['front', 'back', 'left', 'right']):
        return jsonify({'error': 'Missing one or more views'}), 400

    results = []
    image_paths = []
    
    for view in ['front', 'back', 'left', 'right']:
        file = request.files[view]
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{view}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_paths.append(filepath)
            
            # Prediction
            processed_img = preprocess_image(filepath)
            pred = model.predict(processed_img)[0][0]
            is_defective = pred > 0.5
            confidence = pred * 100 if is_defective else (1 - pred) * 100
            
            # Defect detection
            defects = detect_defects(filepath) if is_defective else []
            
            # Save visualization
            vis_path = f"static/uploads/vis_{filename}"
            if is_defective:
                img = cv2.imread(filepath)
                for defect in defects:
                    cv2.rectangle(img, (defect['x'], defect['y']), 
                                 (defect['x']+10, defect['y']+10), (0, 0, 255), 2)
                cv2.imwrite(vis_path, img)
            
            results.append({
                'view': view.capitalize(),
                'prediction': 'Defective' if is_defective else 'Normal',
                'confidence': float(confidence),
                'visualization': vis_path if is_defective else None,
                'defects': defects
            })

    # Generate combined visualization (simplified example)
    combined_path = "static/uploads/combined.jpg"
    # [Add your code to combine images here]
    
    return jsonify({
        'views': results,
        'overall_status': 'Defective' if any(r['prediction'] == 'Defective' for r in results) else 'Normal',
        'combined_visualization': combined_path,
        'defect_locations': [d for r in results for d in r['defects']]
    })

if __name__ == '__main__':
    app.run()
