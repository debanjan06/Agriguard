import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Fix deterministic algorithm issue
torch.backends.cudnn.deterministic = False
torch.use_deterministic_algorithms(False)

class AgriDataset(Dataset):
    def __init__(self, data, features_scaler=None, target_col='health_status', mode='train'):
        self.data = data.copy()
        self.mode = mode
        self.target_col = target_col
        
        self.spectral_features = ['NDVI', 'EVI', 'SAVI', 'REP']
        self.weather_features = ['temp_max', 'temp_min', 'humidity', 'rainfall']
        self.all_features = self.spectral_features + self.weather_features
        
        if features_scaler is None:
            self.features_scaler = StandardScaler()
            self.data[self.all_features] = self.features_scaler.fit_transform(self.data[self.all_features])
        else:
            self.features_scaler = features_scaler
            self.data[self.all_features] = self.features_scaler.transform(self.data[self.all_features])
        
        if mode == 'train':
            self.label_encoder = LabelEncoder()
            self.data['encoded_label'] = self.label_encoder.fit_transform(self.data[target_col])
        else:
            self.label_encoder = None
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        spectral = torch.tensor([row[feat] for feat in self.spectral_features], dtype=torch.float32)
        weather = torch.tensor([row[feat] for feat in self.weather_features], dtype=torch.float32)
        spatial_patch = self.create_spatial_representation(row, spectral)
        
        if 'encoded_label' in self.data.columns:
            label = torch.tensor(row['encoded_label'], dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)
            
        return {
            'spatial': spatial_patch,
            'spectral': spectral,
            'weather': weather,
            'label': label
        }
    
    def create_spatial_representation(self, row, spectral_values):
        patch_size = 32
        n_bands = 4
        base_reflectance = torch.tensor([0.06, 0.10, 0.04, 0.50])
        
        ndvi_val = spectral_values[0].item()
        health_factor = max(0.3, (ndvi_val + 1) / 2)
        
        patch = torch.zeros(n_bands, patch_size, patch_size)
        
        for band_idx, base_val in enumerate(base_reflectance):
            spatial_noise = torch.randn(patch_size, patch_size) * 0.02
            health_variation = base_val * health_factor
            
            if health_factor < 0.7:
                disease_pattern = self.generate_disease_pattern(patch_size, 1 - health_factor)
                health_variation = health_variation * (1 - disease_pattern * 0.4)
            
            patch[band_idx] = health_variation + spatial_noise
            patch[band_idx] = torch.clamp(patch[band_idx], 0.01, 0.95)
        
        return patch
    
    def generate_disease_pattern(self, size, severity):
        pattern = torch.zeros(size, size)
        n_spots = max(1, int(severity * 4))  # Reduced complexity
        
        for _ in range(n_spots):
            center_y = torch.randint(5, size-5, (1,)).item()
            center_x = torch.randint(5, size-5, (1,)).item()
            radius = torch.randint(2, 6, (1,)).item()
            
            y_indices, x_indices = torch.meshgrid(
                torch.arange(max(0, center_y-radius), min(size, center_y+radius+1)),
                torch.arange(max(0, center_x-radius), min(size, center_x+radius+1)),
                indexing='ij'
            )
            
            distances = torch.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
            mask = distances <= radius
            
            intensity = torch.clamp(1 - distances / radius, 0, 1)
            pattern[y_indices[mask], x_indices[mask]] = torch.maximum(
                pattern[y_indices[mask], x_indices[mask]], 
                intensity[mask] * severity
            )
        
        return pattern

class MultiModalCNN(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # Simplified spatial encoder - replaced AdaptiveAvgPool2d
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),  # Fixed size calculation
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Spectral encoder
        self.spectral_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        
        # Weather encoder
        self.weather_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
            
    def forward(self, spatial, spectral, weather):
        spatial_features = self.spatial_encoder(spatial)
        spectral_features = self.spectral_encoder(spectral)
        weather_features = self.weather_encoder(weather)
        
        combined_features = torch.cat([spatial_features, spectral_features, weather_features], dim=1)
        output = self.fusion_layer(combined_features)
        
        return output
    
    def training_step(self, batch, batch_idx):
        spatial = batch['spatial']
        spectral = batch['spectral']
        weather = batch['weather']
        labels = batch['label']
        
        outputs = self(spatial, spectral, weather)
        loss = self.criterion(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        spatial = batch['spatial']
        spectral = batch['spectral']
        weather = batch['weather']
        labels = batch['label']
        
        outputs = self(spatial, spectral, weather)
        loss = self.criterion(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def create_dataloaders(train_df, val_df, batch_size=8):
    train_dataset = AgriDataset(train_df, mode='train')
    
    val_dataset = AgriDataset(val_df, features_scaler=train_dataset.features_scaler, mode='val')
    val_dataset.label_encoder = train_dataset.label_encoder
    val_dataset.data['encoded_label'] = train_dataset.label_encoder.transform(val_dataset.data['health_status'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, train_dataset.label_encoder

def train_working_model(csv_path='data/processed/agriguard_training_dataset.csv'):
    print("Starting Working AgriGuard Training")
    print("=" * 50)
    
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['health_status'], random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    train_loader, val_loader, label_encoder = create_dataloaders(train_df, val_df, batch_size=8)
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"Batch shapes - Spatial: {batch['spatial'].shape}, Spectral: {batch['spectral'].shape}")
    
    num_classes = len(label_encoder.classes_)
    model = MultiModalCNN(num_classes=num_classes, learning_rate=1e-3)
    
    print(f"Model initialized with {num_classes} classes: {list(label_encoder.classes_)}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(batch['spatial'], batch['spectral'], batch['weather'])
        print(f"Forward pass successful - Output shape: {output.shape}")
    
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        deterministic=False  # Disable deterministic mode
    )
    
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'hyperparameters': model.hparams
    }, 'models/agriguard_working_model.pth')
    
    print("Training completed and model saved to models/agriguard_working_model.pth!")
    return model, label_encoder

# Run the training
if __name__ == "__main__":
    # Disable deterministic algorithms
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model, label_encoder = train_working_model()
    
    if model is not None:
        print("\n" + "=" * 50)
        print("SUCCESS! AgriGuard CNN Training Completed")
        print("=" * 50)
        
        # Test final model
        model.eval()
        spatial_test = torch.randn(1, 4, 32, 32)
        spectral_test = torch.randn(1, 4)
        weather_test = torch.randn(1, 4)
        
        with torch.no_grad():
            output = model(spatial_test, spectral_test, weather_test)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)
            
        print(f"Final test prediction: {label_encoder.inverse_transform(pred.numpy())[0]}")
        print(f"Confidence: {probs.max().item():.3f}")
