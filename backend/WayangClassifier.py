# backend/WayangClassifier.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class TrainingConfig:
    def __init__(self):
        self.min_accuracy = 80.0    # Minimal acceptable
        self.target_accuracy = 90.0 # Target ideal  
        self.max_accuracy = 95.0    # Saturation point
        self.max_epochs = 100       # Maximum training epochs
        
    def should_continue_training(self, current_accuracy, epoch, current_val_accuracy=None):
        """Decision logic untuk early stopping"""
        accuracy = current_val_accuracy if current_val_accuracy is not None else current_accuracy
        
        if accuracy >= self.target_accuracy:
            print(f"✅ Target accuracy {self.target_accuracy}% tercapai! Stopping training.")
            return False
        elif epoch >= self.max_epochs and accuracy < self.min_accuracy:
            print(f"⚠️  Epoch maksimal tercapai dengan accuracy {accuracy:.2f}% di bawah minimum {self.min_accuracy}%")
            return False
        elif accuracy >= self.max_accuracy:
            print(f"🎯 Accuracy sudah jenuh di {accuracy:.2f}%")
            return False
        else:
            return True

class WayangClassifier:
    def __init__(self):
        self.model = None
        self.wayang_classes = ['GARENG', 'SEMAR', 'PETRUK', 'BAGONG']
        self.training_config = TrainingConfig()
        self.history = None
    
    def load_data(self, base_path='dataset/', img_size=128, grayscale=False):
        """Load wayang images from folders - improved version"""
        images = []
        labels = []
        
        wayang_folders = {
            'gareng': 0,
            'semar': 1, 
            'petruk': 2,
            'bagong': 3
        }
        
        print(f"📁 Loading data from: {base_path}")
        print(f"🖼️  Image size: {img_size}x{img_size}, Grayscale: {grayscale}")
        
        for wayang_name, label in wayang_folders.items():
            folder_path = os.path.join(base_path, wayang_name)
            
            if not os.path.exists(folder_path):
                print(f"❌ Folder not found: {folder_path}")
                continue
                
            all_files = os.listdir(folder_path)
            image_files = [os.path.join(folder_path, f) for f in all_files 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"🖼️  Loading {len(image_files)} images from {wayang_name}...")
            
            for file_path in image_files:
                try:
                    if grayscale:
                        image = Image.open(file_path).convert('L')  # Grayscale
                    else:
                        image = Image.open(file_path).convert('RGB')  # Color
                    
                    image = image.resize((img_size, img_size))
                    image_array = np.array(image) / 255.0
                    images.append(image_array)
                    labels.append(label)
                    
                except Exception as e:
                    print(f"  ❌ Error loading {file_path}: {e}")
        
        if len(images) == 0:
            return None, None
        
        if grayscale:
            X = np.array(images).reshape(-1, img_size, img_size, 1)
        else:
            X = np.array(images)  # Shape: (samples, 128, 128, 3)
        
        y = np.array(labels)
        
        print(f"🎉 Dataset loaded: {X.shape[0]} images, Shape: {X.shape}")
        return X, y
    
    def create_model(self, input_shape=(128, 128, 3)):
        """Create CNN model for color images"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            keras.layers.MaxPooling2D(2,2),
            
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            
            keras.layers.Conv2D(128, (3,3), activation='relu'), 
            keras.layers.MaxPooling2D(2,2),
            
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(4, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        print(f"✅ CNN Model created for input shape: {input_shape}")
        return model
    
    def prepare_data(self, X, y, test_size=0.2, validation_size=0.2):
        """Split data into training, validation and test sets"""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        val_ratio = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"📊 Data Split:")
        print(f"   Training: {X_train.shape[0]} images")
        print(f"   Validation: {X_val.shape[0]} images") 
        print(f"   Test: {X_test.shape[0]} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100):
        """Train the model dengan early stopping based on accuracy target"""
        if self.model is None:
            self.model = self.create_model(input_shape=X_train.shape[1:])
        
        self.training_config.max_epochs = epochs
        
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        print(f"🎯 Starting training with target accuracy: {self.training_config.target_accuracy}%")
        print(f"   Min accuracy: {self.training_config.min_accuracy}%")
        print(f"   Max epochs: {self.training_config.max_epochs}")
        
        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            epoch_history = self.model.fit(
                X_train, y_train,
                epochs=1,
                batch_size=32,
                validation_data=(X_val, y_val),
                verbose=1
            )
            
            current_accuracy = epoch_history.history['accuracy'][0] * 100
            current_val_accuracy = epoch_history.history['val_accuracy'][0] * 100
            
            history['accuracy'].append(epoch_history.history['accuracy'][0])
            history['val_accuracy'].append(epoch_history.history['val_accuracy'][0])
            history['loss'].append(epoch_history.history['loss'][0])
            history['val_loss'].append(epoch_history.history['val_loss'][0])
            
            print(f"   Training Accuracy: {current_accuracy:.2f}%")
            print(f"   Validation Accuracy: {current_val_accuracy:.2f}%")
            
            if not self.training_config.should_continue_training(current_accuracy, epoch + 1, current_val_accuracy):
                print(f"🛑 Training stopped at epoch {epoch + 1}")
                break
        
        self.history = history
        self.model.save('wayang_classifier.h5')
        print("✅ Model trained and saved!")
        
        return history
    
    def predict(self, image_file, img_size=128, grayscale=False):
        """Predict wayang from image file"""
        if self.model is None:
            self.model = keras.models.load_model('wayang_classifier.h5')
        
        # Preprocess image
        image = Image.open(io.BytesIO(image_file))
        
        if grayscale:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        
        # RESIZE dulu!
        image = image.resize((img_size, img_size))
        image_array = np.array(image) / 255.0
        
        # Baru reshape
        if grayscale:
            image_array = image_array.reshape(1, img_size, img_size, 1)
        else:
            image_array = image_array.reshape(1, img_size, img_size, 3)
        
        # Predict
        prediction = self.model.predict(image_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return {
            'wayang': self.wayang_classes[predicted_class],
            'confidence': float(confidence * 100),
            'all_predictions': dict(zip(self.wayang_classes, prediction[0]))
        }

# USAGE EXAMPLE:
if __name__ == "__main__":
    # Initialize classifier
    classifier = WayangClassifier()
    
    # Load and prepare data
    X, y = classifier.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(X, y)
    
    # Train model dengan accuracy target
    history = classifier.train(X_train, y_train, X_val, y_val, epochs=100)
    
    # Evaluate on test set
    classifier.evaluate_model(X_test, y_test)