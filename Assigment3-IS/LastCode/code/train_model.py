import os
import warnings

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# ========== Suppress Warnings ==========
warnings.filterwarnings("ignore")


# ========== Dataset Class (Top‐Level!) ==========
class QuranDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, img_size=224):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
        self.valid_files = [
            f for f in dataframe['filename']
            if os.path.exists(os.path.join(img_dir, f))
        ]

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        filename = self.valid_files[idx]
        img_path = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            label = int(self.data.loc[self.data['filename'] == filename, 'label'])
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            dummy = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            return self.transform(dummy), 0


def main():
    # === Config ===
    IMG_DIR     = r"G:\Group Project\PY CODE\FINALISE CODE\data\quran_images"
    LABEL_CSV   = r"G:\Group Project\PY CODE\FINALISE CODE\data\quran_labels.csv"
    MODEL_DIR   = r"G:\Group Project\PY CODE\FINALISE CODE\data"
    BATCH_SIZE  = 32
    EPOCHS      = 50
    LR          = 3e-4
    IMG_SIZE    = 224
    NUM_WORKERS = 0    # Windows: set to 0
    MIXED_PREC  = True

    os.makedirs(MODEL_DIR, exist_ok=True)

    # === Device & AMP Scaler ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(enabled=MIXED_PREC)

    print(f"\n🚀 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
    print(f"   Mixed Precision: {MIXED_PREC}\n")

    # === Path Checks ===
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"Image dir not found: {IMG_DIR}")
    if not os.path.isfile(LABEL_CSV):
        raise FileNotFoundError(f"Label CSV not found: {LABEL_CSV}")
    print("✅ Paths validated\n")

    # === Load & Encode Labels ===
    df = pd.read_csv(LABEL_CSV)
    print(f"📊 Loaded {len(df)} entries")
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['surah_name'])
    n_classes = len(le.classes_)
    print(f"   → {n_classes} unique classes\n")

    # === Train/Val Split ===
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    print(f"📈 Train: {len(train_df)} | 🧪 Val: {len(val_df)}\n")

    # === Weighted Sampler ===
    counts = train_df['label'].value_counts().sort_index().tolist()
    class_w = 1. / torch.tensor(counts, dtype=torch.float)
    sample_w = class_w[train_df['label'].values]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    # === Transforms ===
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE+20, IMG_SIZE+20)),
        transforms.RandomRotation(5),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ColorJitter(0.2,0.2,0,0),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    # === Datasets & DataLoaders ===
    train_ds = QuranDataset(train_df, IMG_DIR, transform=train_tf, img_size=IMG_SIZE)
    val_ds   = QuranDataset(val_df,   IMG_DIR, transform=val_tf,   img_size=IMG_SIZE)
    print(f"   Valid train imgs: {len(train_ds)}/{len(train_df)}")
    print(f"   Valid  val imgs: {len(val_ds)}/{len(val_df)}\n")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # === Model Setup ===
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model = model.to(device)
    print(f"🧠 Model ready → {n_classes} classes\n")

    # === Loss, Optimizer & Scheduler ===
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3
    )

    # === Training Loop ===
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for i, (imgs, lbls) in enumerate(train_loader, 1):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with autocast(enabled=MIXED_PREC):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            if i % 10 == 0:
                print(f" Epoch {epoch}/{EPOCHS} — Batch {i}/{len(train_loader)} — Loss {loss.item():.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total   = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                correct += (preds == lbls).sum().item()
                total   += lbls.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss   = val_loss     / len(val_loader.dataset)
        val_acc    = correct      / total
        scheduler.step(val_acc)

        print(f"\n→ Epoch {epoch} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Val Accuracy: {val_acc*100:.2f}%\n")

        # --- Save best ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(MODEL_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': le,
                'classes': le.classes_
            }, best_path)
            print(f"💾 New best model saved to {best_path}\n")

    # === Final checkpoint ===
    final_path = os.path.join(MODEL_DIR, "best_model_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': le.classes_.tolist(),
        'input_size': IMG_SIZE,
        'arch': 'resnet50'
    }, final_path)
    print(f"🏁 Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    # Windows multiprocessing safety
    mp.freeze_support()
    main()
