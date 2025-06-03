import torch
import torch.nn as nn
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------
# 1. Model Definition (Must match training architecture)
# --------------------

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        # Lightweight custom CNN
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Transformations
emotion_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --------------------
# 2. Inference Pipeline
# --------------------

class FaceEmotionDetector:
    def __init__(self, model_path='emotion_model.pth'):
        # Emotion mapping
        self.emotion_map = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear',
            3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
        # Initialize model
        self.emotion_model = EmotionCNN().to(device)
        self.emotion_model.load_state_dict(torch.load(model_path, map_location=device))
        self.emotion_model.eval()
        
        # Initialize face detector on CPU to save VRAM
        self.face_detector = MTCNN(keep_all=True, device='cpu')
        
    def detect(self, image_path, confidence_threshold=0.85):
        """Detect faces and emotions in an image"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
        
        # Detect faces
        boxes, probs = self.face_detector.detect(image)
        results = []
        
        if boxes is not None:
            draw = ImageDraw.Draw(original_image)
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < confidence_threshold:
                    continue
                    
                # Extract face ROI
                x1, y1, x2, y2 = box
                face = image.crop((x1, y1, x2, y2))
                
                # Convert to grayscale and transform
                face_gray = face.convert('L')
                emotion_tensor = emotion_transform(face_gray).unsqueeze(0)
                
                # Emotion detection
                with torch.no_grad():
                    emotion_tensor = emotion_tensor.to(device)
                    output = self.emotion_model(emotion_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    _, predicted = torch.max(output.data, 1)
                    emotion = self.emotion_map[predicted.item()]
                    confidence = probabilities[predicted.item()].item()
                
                # Draw bounding box and label
                draw.rectangle(box.tolist(), outline='red', width=2)
                draw.text((x1, y1 - 25), f"{emotion} ({confidence:.2f})", fill='red')
                
                results.append({
                    'box': box,
                    'emotion': emotion,
                    'confidence': confidence
                })
        
        return original_image, results

# --------------------
# 3. Main Prediction Workflow
# --------------------

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Face Emotion Detection')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default='emotion_model.pth', 
                        help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.85,
                        help='Confidence threshold for face detection')
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file not found - {args.image}")
        return
        
    if not os.path.exists(args.model):
        print(f"Error: Model file not found - {args.model}")
        return
    
    # Initialize detector
    print("Loading emotion detection model...")
    detector = FaceEmotionDetector(args.model)
    
    # Process image
    print(f"Processing image: {args.image}")
    result_image, results = detector.detect(args.image, args.threshold)
    
    # Display results
    plt.figure(figsize=(12, 10))
    plt.imshow(result_image)
    plt.title('Face Emotion Detection')
    plt.axis('off')
    plt.show()
    
    # Print results
    print("\nDetection Results:")
    for i, res in enumerate(results):
        print(f"Face {i+1}:")
        print(f"  Position: ({res['box'][0]:.1f}, {res['box'][1]:.1f}) to ({res['box'][2]:.1f}, {res['box'][3]:.1f})")
        print(f"  Emotion: {res['emotion']} (Confidence: {res['confidence']:.2f})")
    
    # Memory cleanup
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()