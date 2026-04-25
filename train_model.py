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
import time

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'remote_sensing_cd.settings')
django.setup()

from datasets.models import Dataset as DBDataset, ModelConfiguration, TrainingLog
from change_detection.ml_models.lenet_model import LENet
from admin_panel.models import SystemLog

class ChangeDetectionDataset(Dataset):
    """Custom dataset for change detection"""
    def __init__(self, image_pairs, labels, target_size=(256, 256)):
        self.image_pairs = image_pairs
        self.labels = labels
        self.target_size = target_size
        
        # Image transformations
        self.img_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Label transformation - resize to match output
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
            # Load images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            
            # Apply transformations
            img1 = self.img_transform(img1)
            img2 = self.img_transform(img2)
            label = self.label_transform(label)
            
            # Convert label to binary (0 or 1) and remove channel dimension
            label = (label > 0.5).long().squeeze(0)
            
            return img1, img2, label
            
        except Exception as e:
            print(f"Error loading {img1_path}: {e}")
            # Return dummy data
            return torch.zeros(3, *self.target_size), torch.zeros(3, *self.target_size), torch.zeros(*self.target_size).long()

def load_dataset(dataset_path):
    """Load image pairs and labels from dataset path"""
    image_pairs = []
    labels = []
    
    # Check different possible structures
    a_path = os.path.join(dataset_path, 'A')
    b_path = os.path.join(dataset_path, 'B')
    label_path = os.path.join(dataset_path, 'label')
    
    if not os.path.exists(a_path):
        print(f"Warning: Path {a_path} does not exist")
        return [], []
    
    print(f"Loading from: {dataset_path}")
    print(f"  - A: {a_path}")
    print(f"  - B: {b_path}")
    print(f"  - Labels: {label_path}")
    
    # Get all files
    a_files = sorted([f for f in os.listdir(a_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for filename in a_files:
        img1 = os.path.join(a_path, filename)
        img2 = os.path.join(b_path, filename)
        lbl = os.path.join(label_path, filename)
        
        if os.path.exists(img2) and os.path.exists(lbl):
            image_pairs.append((img1, img2))
            labels.append(lbl)
    
    return image_pairs, labels

def calculate_metrics(outputs, labels):
    """Calculate precision, recall, F1, IoU"""
    pred = torch.argmax(outputs, dim=1)
    
    # True Positives, False Positives, False Negatives
    tp = ((pred == 1) & (labels == 1)).sum().float()
    fp = ((pred == 1) & (labels == 0)).sum().float()
    fn = ((pred == 0) & (labels == 1)).sum().float()
    tn = ((pred == 0) & (labels == 0)).sum().float()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    
    return precision.item(), recall.item(), f1.item(), iou.item()

def train_model(config_id, dataset_id):
    """Main training function"""
    # Load configuration
    config = ModelConfiguration.objects.get(id=config_id)
    dataset = DBDataset.objects.get(id=dataset_id)
    
    print(f"Training with configuration: {config.name}")
    print(f"Dataset: {dataset.name} ({dataset.dataset_type})")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    image_pairs, labels = load_dataset(dataset.path)
    
    if len(image_pairs) == 0:
        print("Error: No image pairs found!")
        return
    
    print(f"Loaded {len(image_pairs)} image pairs")
    
    # Split into train and validation (80-20)
    split_idx = int(0.8 * len(image_pairs))
    train_pairs = image_pairs[:split_idx]
    train_labels = labels[:split_idx]
    val_pairs = image_pairs[split_idx:]
    val_labels = labels[split_idx:]
    
    print(f"Training samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    
    # Create datasets and dataloaders
    train_dataset = ChangeDetectionDataset(train_pairs, train_labels)
    val_dataset = ChangeDetectionDataset(val_pairs, val_labels)
    
    # Reduce workers to 0 to avoid the warning (or set to 2 max)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=0)
    
    # Initialize model
    model = LENet(in_channels=3, num_classes=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    SystemLog.objects.create(
        log_type='info',
        message=f'Started training: {config.name}',
        module='model_training'
    )
    
    # Set training start time
    from django.utils import timezone
    config.is_training = True
    config.training_started_at = timezone.now()
    config.save()
    
    best_f1 = 0.0
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
        for batch_idx, (img1, img2, labels_batch) in enumerate(pbar):
            img1, img2, labels_batch = img1.to(device), img2.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(img1, img2)
            
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate metrics
            with torch.no_grad():
                p, r, f, i = calculate_metrics(outputs, labels_batch)
                train_metrics['precision'] += p
                train_metrics['recall'] += r
                train_metrics['f1'] += f
                train_metrics['iou'] += i
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'f1': f"{f:.4f}"
            })
        
        epoch_loss = running_loss / len(train_loader)
        train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Val]')
            for img1, img2, labels_batch in pbar:
                img1, img2, labels_batch = img1.to(device), img2.to(device), labels_batch.to(device)
                
                outputs = model(img1, img2)
                loss = criterion(outputs, labels_batch)
                
                val_loss += loss.item()
                
                p, r, f, i = calculate_metrics(outputs, labels_batch)
                val_metrics['precision'] += p
                val_metrics['recall'] += r
                val_metrics['f1'] += f
                val_metrics['iou'] += i
        
        val_loss /= len(val_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.epochs} Summary:")
        print(f"  Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val IoU: {val_metrics['iou']:.4f}")
        
        # Save training log
        TrainingLog.objects.create(
            configuration=config,
            dataset=dataset,
            epoch=epoch + 1,
            train_loss=epoch_loss,
            val_loss=val_loss,
            precision=val_metrics['precision'],
            recall=val_metrics['recall'],
            f1_score=val_metrics['f1'],
            iou=val_metrics['iou']
        )
        
        # Update configuration with current training status
        config.current_epoch = epoch + 1
        config.current_train_loss = epoch_loss
        config.current_val_loss = val_loss
        config.current_f1 = val_metrics['f1']
        config.is_training = True
        config.save()
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_path = f'trained_models/weights/lenet_{config.name}_best.pth'
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved (F1: {best_f1:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'trained_models/weights/lenet_{config.name}_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint saved: epoch {epoch+1}")
        
        scheduler.step()
    
    # Save final model
    final_path = f'trained_models/weights/lenet_{config.name}_final.pth'
    torch.save(model.state_dict(), final_path)
    config.weights_path = final_path
    config.is_training = False
    config.save()
    
    SystemLog.objects.create(
        log_type='info',
        message=f'Training completed: {config.name} - Best F1: {best_f1:.4f}',
        module='model_training'
    )
    
    print(f'\n{"="*60}')
    print(f'Training completed!')
    print(f'Best F1 Score: {best_f1:.4f}')
    print(f'Final model: {final_path}')
    print(f'Best model: {best_path}')
    print(f'{"="*60}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LENet model')
    parser.add_argument('--config_id', type=int, required=True, help='Configuration ID')
    parser.add_argument('--dataset_id', type=int, required=True, help='Dataset ID')
    
    args = parser.parse_args()
    
    try:
        train_model(args.config_id, args.dataset_id)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
