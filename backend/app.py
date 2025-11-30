# backend/app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from WayangClassifier import WayangClassifier
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Initialize classifier
print("🚀 Loading Wayang Classifier Model...")
classifier = WayangClassifier()

try:
    # Load model yang sudah ditrained
    classifier.model = classifier.create_model()
    classifier.model.load_weights('wayang_classifier.h5')
    print("✅ Pre-trained model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("💡 Please train the model first using train_model.py")

@app.route('/')
def home():
    return jsonify({
        "message": "Wayang Classification API", 
        "status": "active",
        "classes": classifier.wayang_classes
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Use the predict method from your class
        result = classifier.predict(file.read())
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)