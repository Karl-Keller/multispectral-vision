import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import json
from pathlib import Path
from PIL import Image
import numpy as np
from pycocotools.coco import COCO

class MultiSpectralDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load COCO annotations
        ann_file = self.data_dir / f'_annotations.coco.json'
        self.coco = COCO(ann_file)
        
        # Get image IDs for this split
        self.img_ids = list(self.coco.imgs.keys())
        
        # Get category information
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.num_classes = len(self.categories)
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        
        # Load image
        img_path = self.data_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create segmentation mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.int64)
        for ann in anns:
            # For each annotation, fill the corresponding area with category ID
            if 'segmentation' in ann:
                mask_ann = self.coco.annToMask(ann)
                mask[mask_ann == 1] = ann['category_id']
        
        # Convert to tensors
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask

def create_model(num_classes):
    # Load DeepLabV3+ with ResNet50 backbone
    model = deeplabv3_resnet50(pretrained=True)
    
    # Modify the classifier for our number of classes
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for images, masks in train_loader:
            if device == 'cuda':
                images = images.cuda()
                masks = masks.cuda()
            
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                if device == 'cuda':
                    images = images.cuda()
                    masks = masks.cuda()
                
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
    
    return model

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Data directories
    data_dir = Path(__file__).parent.parent / 'data' / 'dataset'
    
    # Create datasets
    train_dataset = MultiSpectralDataset(data_dir, split='train')
    val_dataset = MultiSpectralDataset(data_dir, split='valid')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Create model
    model = create_model(train_dataset.num_classes)
    
    # Train model
    model = train_model(model, train_loader, val_loader, num_epochs=10, device=device)
    
    # Save model
    torch.save(model.state_dict(), 'multispectral_model.pth')

if __name__ == '__main__':
    main()