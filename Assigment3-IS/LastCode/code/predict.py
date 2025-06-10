import os
import argparse
import warnings
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# ========== Suppress Warnings ==========
warnings.filterwarnings("ignore")

# ========== Configuration ==========
MODEL_PATH = r"G:\Group Project\PY CODE\FINALISE CODE\data\best_model_final.pth"
IMG_SIZE = 224
CLASS_NAMES = None  # Will be loaded from model checkpoint

# ========== Load Model Function ==========
def load_model(model_path):
    """Load trained model with checkpoint data"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Get class names
    global CLASS_NAMES
    CLASS_NAMES = checkpoint.get('classes', [])
    
    # Recreate model architecture
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"   Classes: {len(CLASS_NAMES)}")
    print(f"   Input size: {checkpoint.get('input_size', IMG_SIZE)}")
    
    return model

# ========== Image Transformation ==========
def get_transform(img_size):
    """Validation transforms used during training"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ========== Prediction Function ==========
def predict_image(model, image_path, transform):
    """Run prediction on single image"""
    try:
        # Load and transform image
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Run prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probabilities, 0)
            
        # Get class name
        class_name = CLASS_NAMES[pred_idx.item()] if CLASS_NAMES else f"Class {pred_idx.item()}"
        
        return {
            'class_idx': pred_idx.item(),
            'class_name': class_name,
            'confidence': conf.item(),
            'probabilities': probabilities.numpy(),
            'image': img
        }
    
    except Exception as e:
        print(f"⚠️ Error processing image: {e}")
        return None

# ========== Visualization Function ==========
def visualize_prediction(result, top_k=5):
    """Create visualization of prediction results"""
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Display image
    plt.subplot(2, 1, 1)
    plt.imshow(result['image'])
    plt.title(f"Prediction: {result['class_name']} ({result['confidence']:.2%})")
    plt.axis('off')
    
    # Display probabilities
    plt.subplot(2, 1, 2)
    if CLASS_NAMES:
        sorted_indices = np.argsort(result['probabilities'])[::-1][:top_k]
        sorted_classes = [CLASS_NAMES[i] for i in sorted_indices]
        sorted_probs = result['probabilities'][sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0.3, 1, top_k))
        bars = plt.barh(range(top_k), sorted_probs[::-1], color=colors)
        
        plt.yticks(range(top_k), sorted_classes[::-1])
        plt.xlabel('Probability')
        plt.title('Top Predictions')
        plt.xlim(0, 1)
        
        # Add probability text
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, i, f"{width:.2%}", va='center')
    
    plt.tight_layout()
    return plt

# ========== Main Prediction Workflow ==========
def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Quran Surah Classifier Prediction')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default=MODEL_PATH, 
                        help=f'Path to trained model (default: {MODEL_PATH})')
    parser.add_argument('--topk', type=int, default=3,
                        help='Number of top predictions to show (default: 3)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save output to file instead of showing')
    args = parser.parse_args()
    
    # Check input image
    if not os.path.exists(args.image):
        print(f"Error: Image file not found - {args.image}")
        return
    
    # Load model
    print("\n🚀 Loading classification model...")
    model = load_model(args.model)
    transform = get_transform(IMG_SIZE)
    
    # Run prediction
    print(f"\n🔍 Analyzing image: {args.image}")
    result = predict_image(model, args.image, transform)
    
    if result:
        # Print results
        print("\n📊 Prediction Results:")
        print(f"  Surah: {result['class_name']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Class Index: {result['class_idx']}")
        
        # Visualize
        plt = visualize_prediction(result, args.topk)
        
        # Save or show
        if args.save:
            plt.savefig(args.save, dpi=300, bbox_inches='tight')
            print(f"\n💾 Results saved to: {args.save}")
        else:
            plt.show()
    else:
        print("❌ Prediction failed")

if __name__ == "__main__":
    main()