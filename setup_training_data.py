import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'remote_sensing_cd.settings')
django.setup()

from datasets.models import Dataset as DBDataset, ModelConfiguration
from admin_panel.models import SystemLog

def setup_training_data():
    """Create Dataset and ModelConfiguration records for training"""
    
    print("Setting up training data...")
    
    # Create Dataset
    dataset, created = DBDataset.objects.get_or_create(
        name='Remote Sensing Change Detection Dataset',
        defaults={
            'dataset_type': 'change_detection',
            'description': 'Local dataset for change detection training with Time1 and Time2 images',
            'path': 'dataset_path/data',
            'is_active': True
        }
    )
    
    if created:
        print(f"✓ Created Dataset: ID={dataset.id}, Name='{dataset.name}'")
        SystemLog.objects.create(
            log_type='info',
            message=f'Dataset created: {dataset.name}',
            module='setup'
        )
    else:
        print(f"✓ Dataset already exists: ID={dataset.id}, Name='{dataset.name}'")
    
    # Create ModelConfiguration
    config, created = ModelConfiguration.objects.get_or_create(
        name='Default LENet Configuration',
        defaults={
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 50,
            'csdw_enabled': True,
            'layer_exchange_enabled': True,
            'current_epoch': 0,
            'is_training': False
        }
    )
    
    if created:
        print(f"✓ Created ModelConfiguration: ID={config.id}, Name='{config.name}'")
        SystemLog.objects.create(
            log_type='info',
            message=f'Model configuration created: {config.name}',
            module='setup'
        )
    else:
        print(f"✓ ModelConfiguration already exists: ID={config.id}, Name='{config.name}'")
    
    print("\n" + "="*60)
    print("Setup Complete! You can now train the model with:")
    print(f"python train_model.py --config_id {config.id} --dataset_id {dataset.id}")
    print("="*60)
    
    return dataset.id, config.id

if __name__ == '__main__':
    setup_training_data()
