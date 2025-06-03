import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import warnings
from torch.cuda import amp  # Mixed precision support
from PIL import Image       # ← Add this import
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------
# Dataset Preparation (Optimized for 4GB VRAM)
# --------------------
class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # Preload and cache images in memory
        self.images = []
        for i in range(len(self.data)):
            img_pixels = self.data.iloc[i]['pixels']
            img_array = np.array([int(pixel) for pixel in img_pixels.split()], dtype=np.uint8)
            img = img_array.reshape(48, 48)
            self.images.append(img)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx]).convert('L')  # Grayscale
        label = int(self.data.iloc[idx]['emotion'])
        return self.transform(img) if self.transform else img, label

# --------------------
# Model Architecture (Lightweight)
# --------------------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
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
        return self.classifier(x)

# --------------------
# Training Configuration
# --------------------
def train_emotion_model():
    # Transformations
    emotion_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load dataset
    dataset = FER2013Dataset(csv_file='fer2013.csv', transform=emotion_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Data loaders with reduced batch size
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
    
    # Initialize model
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Mixed precision scaler
    scaler = amp.GradScaler()
    
    # Training loop
    best_accuracy = 0.0
    print("Starting training...")
    for epoch in range(15):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Mixed precision training
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/15], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Update scheduler and save best model
        scheduler.step(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'emotion_model.pth')
            print(f"Saved new best model with accuracy: {best_accuracy:.2f}%")
    
    print(f'Training complete. Best Validation Accuracy: {best_accuracy:.2f}%')

# --------------------
# Main Execution
# --------------------
if __name__ == "__main__":
    # Memory optimization
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    # Train and save model
    train_emotion_model()