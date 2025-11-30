# backend/test_final.py
from WayangClassifier import WayangClassifier
import os

print("🧪 FINAL TEST - Color Model")
classifier = WayangClassifier()

# Load trained model
classifier.model = classifier.create_model(input_shape=(128, 128, 3))
classifier.model.load_weights('wayang_classifier.h5')
print("✅ Model loaded!")

# Test prediction
test_image = 'dataset/gareng/gareng_1.jpg'
if os.path.exists(test_image):
    with open(test_image, 'rb') as f:
        result = classifier.predict(f.read(), img_size=128, grayscale=False)
        print(f"\n🎭 PREDICTION RESULT:")
        print(f"   Wayang: {result['wayang']}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print(f"   Details: {result['all_predictions']}")
else:
    print("❌ Test image not found")