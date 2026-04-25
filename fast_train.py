import os
import sys
import django
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'remote_sensing_cd.settings')
django.setup()

from datasets.models import Dataset as DBDataset, ModelConfiguration, TrainingLog
from change_detection.ml_models.lenet_lite import LENetLite
from admin_panel.models import SystemLog

class FastChangeDetectionDataset(Dataset):
    """Optimized dataset loader"""
    def __init__(self, image_pairs, labels, target_size=(128, 128)):  # Smaller size
        self.image_pairs = image_pairs
        self.labels = labels
        self.target_size = target_size
        
        self.img_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        
        self.label_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        label_path = self.labels[idx]
        
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            
            img1 = self.img_transform(img1)
            img2 = self.img_transform(img2)
            label = self.label_transform(label)
            label = (label > 0.5).long().squeeze(0)
            
            return img1, img2, label
        except:
            return torch.zeros(3, *self.target_size), torch.zeros(3, *self.target_size), torch.zeros(*self.target_size).long()

def load_dataset(dataset_path):
    image_pairs = []
    labels = []
    
    a_path = os.path.join(dataset_path, 'A')
    b_path = os.path.join(dataset_path, 'B')
    label_path = os.path.join(dataset_path, 'label')
    
    if not os.path.exists(a_path):
        return [], []
    
    a_files = sorted([f for f in os.listdir(a_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for filename in a_files:
        img1 = os.path.join(a_path, filename)
        img2 = os.path.join(b_path, filename)
        lbl = os.path.join(label_path, filename)
        
        if os.path.exists(img2) and os.path.exists(lbl):
            image_pairs.append((img1, img2))
            labels.append(lbl)
    
    return image_pairs, labels

def fast_train():
    """Ultra-fast training for testing"""
    
    print("="*60)
    print("FAST TRAINING MODE")
    print("="*60)
    
    # Get dataset
    dataset = DBDataset.objects.filter(is_active=True).first()
    if not dataset:
        print("No dataset found!")
        return
    
    print(f"Dataset: {dataset.name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    image_pairs, labels = load_dataset(dataset.path)
    print(f"Samples: {len(image_pairs)}")
    
    # Use only subset for ultra-fast training
    max_samples = min(50, len(image_pairs))
    image_pairs = image_pairs[:max_samples]
    labels = labels[:max_samples]
    print(f"Using {max_samples} samples for fast training")
    
    # Create dataset
    train_dataset = FastChangeDetectionDataset(image_pairs, labels, target_size=(128, 128))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    
    # Initialize lightweight model
    model = LENetLite(in_channels=3, num_classes=2).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training - only 3 epochs
    epochs = 5
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for img1, img2, labels_batch in pbar:
            img1, img2, labels_batch = img1.to(device), img2.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels_batch).sum().item()
            total += labels_batch.numel()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Save model
    save_path = 'trained_models/weights/lenet_lite_fast.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    print(f"\n{'='*60}")
    print(f"✓ Fast training completed!")
    print(f"✓ Model saved: {save_path}")
    print(f"{'='*60}")
    
    SystemLog.objects.create(
        log_type='info',
        message='Fast training completed',
        module='fast_train'
    )

if __name__ == '__main__':
    fast_train()
