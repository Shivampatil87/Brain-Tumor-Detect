"""
Brain Tumor Detection System — Flask Backend
=============================================
Matches the exact CNN architecture and preprocessing pipeline from:
  "brain-tumor-another-dataset-training-4_1_.ipynb"

Key preprocessing: crop_brain_contour() using OpenCV contour detection
Model: Simple but powerful — ZeroPad → Conv2D(32,7x7) → BN → ReLU → MaxPool×2 → Flatten → Dense(1,sigmoid)
Real results: Val Acc = 94.67% | Test Acc = 95.33% | Dataset = 3000 MRI images
"""

import os
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64

# ─── App Configuration ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'neuroscan-secret-2024')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ─── Model Loading ─────────────────────────────────────────────────────────────
model = None

def load_model():
    """Load the trained CNN model (.h5). Gracefully falls back to demo mode."""
    global model
    model_path = os.path.join('model', 'brain_tumor_model.h5')
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded: {model_path}")
            print(f"   Input shape : {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
        except Exception as e:
            print(f"⚠️  Model load failed: {e}")
            model = None
    else:
        print("⚠️  No model found at model/brain_tumor_model.h5")
        print("   Running in DEMO MODE. Run train_model.py to train the model.")

# ─── Helper Functions ──────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def crop_brain_contour(image_array):
    """
    Replicates the notebook's crop_brain_contour() preprocessing step.

    Algorithm:
      1. Convert RGB → Grayscale
      2. Gaussian blur (5×5) to reduce noise
      3. Binary threshold at pixel value 45
      4. Morphological erosion + dilation to clean edges
      5. Find external contours, take the largest (the brain)
      6. Find 4 extreme points (left, right, top, bottom)
      7. Crop image to bounding box of those extreme points

    This isolates the brain tissue and removes black padding,
    improving model accuracy significantly.
    """
    try:
        import cv2
        import imutils

        # Convert PIL-style RGB array to BGR for OpenCV
        image_bgr = image_array[:, :, ::-1].copy()

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        extLeft  = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop   = tuple(c[c[:, :, 1].argmin()][0])
        extBot   = tuple(c[c[:, :, 1].argmax()][0])

        cropped = image_array[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        return cropped

    except Exception:
        # If contour detection fails (e.g. blank image), return original
        return image_array


def preprocess_image(image_path):
    """
    Full preprocessing pipeline matching the notebook exactly:
      1. Open image, convert to RGB
      2. Convert to numpy array (uint8)
      3. crop_brain_contour() — OpenCV contour crop
      4. Resize to 240×240
      5. Normalize to [0, 1]
      6. Add batch dimension → shape (1, 240, 240, 3)
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)                          # uint8, HxWx3

    img_array = crop_brain_contour(img_array)          # Crop to brain

    # Resize with PIL after converting back
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    img_pil = img_pil.resize((240, 240), Image.BICUBIC)

    img_array = np.array(img_pil) / 255.0             # Normalize
    img_array = np.expand_dims(img_array, axis=0)     # (1, 240, 240, 3)
    return img_array


def predict_tumor(image_path):
    """
    Run model inference. Returns dict with label, confidence, has_tumor.

    Model output is a single sigmoid neuron:
      > 0.5  →  Tumor Detected
      ≤ 0.5  →  No Tumor Detected

    Real performance (from notebook):
      Validation Accuracy : 94.67%
      Test Accuracy       : 95.33%
    """
    if model is None:
        # Demo mode — simulated result
        import random
        rng = random.Random()
        prob_tumor = round(rng.uniform(55, 98), 2)
        has_tumor = prob_tumor >= 50
        return {
            'label': 'Tumor Detected' if has_tumor else 'No Tumor Detected',
            'confidence': prob_tumor if has_tumor else round(100 - prob_tumor, 2),
            'prob_tumor': prob_tumor,
            'prob_no_tumor': round(100 - prob_tumor, 2),
            'has_tumor': has_tumor,
            'demo_mode': True
        }

    img_array = preprocess_image(image_path)
    # Model output: shape (1,1), sigmoid
    raw = model.predict(img_array, verbose=0)
    prob_tumor    = float(raw[0][0]) * 100
    prob_no_tumor = 100 - prob_tumor
    has_tumor     = prob_tumor >= 50

    return {
        'label': 'Tumor Detected' if has_tumor else 'No Tumor Detected',
        'confidence': round(prob_tumor if has_tumor else prob_no_tumor, 2),
        'prob_tumor': round(prob_tumor, 2),
        'prob_no_tumor': round(prob_no_tumor, 2),
        'has_tumor': has_tumor,
        'demo_mode': False
    }


def encode_image_base64(image_path):
    """Return base64 data-URI string for embedding in HTML."""
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    ext  = image_path.rsplit('.', 1)[-1].lower()
    mime = 'jpeg' if ext == 'jpg' else ext
    return f"data:image/{mime};base64,{data}"


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/detect')
def detect():
    return render_template('detect.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload, validate, preprocess, predict, return results."""
    if 'file' not in request.files:
        return render_template('detect.html', error="No file included in request.")

    file = request.files['file']

    if file.filename == '':
        return render_template('detect.html', error="No file selected. Please choose an MRI image.")

    if not allowed_file(file.filename):
        return render_template('detect.html',
                               error="Invalid file type. Please upload a JPG or PNG image.")

    # Save securely with UUID to prevent collisions
    filename    = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path   = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(save_path)

    try:
        result    = predict_tumor(save_path)
        image_b64 = encode_image_base64(save_path)
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return render_template('detect.html', error=f"Analysis failed: {str(e)}")

    return render_template('result.html', result=result, image_data=image_b64)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for programmatic access."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG or PNG.'}), 400

    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{filename}")
    file.save(save_path)

    try:
        result = predict_tumor(save_path)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


# ─── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return render_template('detect.html', error="File too large. Maximum allowed size is 16MB."), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('home.html'), 404


# ─── Startup ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
