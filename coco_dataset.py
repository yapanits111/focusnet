# coco_dataset.py
# Dataset loader for FocusNet road hazard detection
# Handles  specific Roboflow export format with split-based COCO annotations
# 
#  Dataset Structure:
# dataset/
# ├── train/
# │   ├── _annotations.coco.json  ← COCO annotations for training
# │   └── [training images]       ← Training images directly here
# ├── valid/
# │   ├── _annotations.coco.json  ← COCO annotations for validation  
# │   └── [validation images]     ← Validation images directly here
# └── test/
#     ├── _annotations.coco.json  ← COCO annotations for testing
#     └── [test images]           ← Test images directly here

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import zipfile
import shutil

def extract_dataset_zip(zip_path, extract_to="/content/dataset", force_extract=False):
    """
    Extract dataset ZIP file from Google Drive for FocusNet training
    
    Args:
        zip_path (str): Path to the dataset ZIP file in Google Drive
        extract_to (str): Directory to extract the dataset to
        force_extract (bool): Re-extract even if directory exists
    
    Returns:
        str: Path to extracted dataset directory
    """
    print(f"📦 Extracting dataset from: {zip_path}")
    print(f"📁 Extract location: {extract_to}")
    
    # Check if already extracted
    if os.path.exists(extract_to) and not force_extract:
        print(f"✅ Dataset already extracted at: {extract_to}")
        return extract_to
    
    # Extract ZIP file
    if force_extract and os.path.exists(extract_to):
        shutil.rmtree(extract_to)
        print("🗑️ Removed existing dataset directory")
    
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print("🔄 Extracting files...")
            zip_ref.extractall(extract_to)
        
        print("✅ Dataset extracted successfully!")
        return extract_to
        
    except Exception as e:
        print(f"❌ Error extracting dataset: {e}")
        raise e

def verify_dataset_structure(dataset_path):
    """
    Verify specific Roboflow export format:
    - train/, valid/, test/ folders
    - _annotations.coco.json in each folder
    - Images directly in each folder
    """
    print("🔍 Verifying dataset structure...")
    
    dataset_contents = os.listdir(dataset_path)
    print(f"📁 Dataset contents: {dataset_contents}")
    
    # Check for required splits
    required_splits = ['train', 'valid', 'test']
    missing_splits = []
    
    for split in required_splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            missing_splits.append(split)
            continue
            
        split_contents = os.listdir(split_path)
        
        # Check for COCO annotation file
        coco_file = '_annotations.coco.json'
        if coco_file not in split_contents:
            print(f"❌ Missing {split}/{coco_file}")
            return False
            
        # Verify annotation file is valid COCO format
        ann_path = os.path.join(split_path, coco_file)
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            required_keys = ['images', 'annotations', 'categories']
            if not all(key in data for key in required_keys):
                print(f"❌ Invalid COCO format in {split}/{coco_file}")
                return False
                
            # Count images and annotations
            img_count = len([f for f in split_contents if f.endswith(('.jpg', '.jpeg', '.png'))])
            ann_count = len(data['annotations'])
            
            print(f"✅ {split}: {img_count} images, {ann_count} annotations")
            
        except Exception as e:
            print(f"❌ Error reading {split}/{coco_file}: {e}")
            return False
    
    if missing_splits:
        print(f"❌ Missing splits: {missing_splits}")
        return False
    
    print("✅ Dataset structure verified!")
    return True

def convert_to_unified_coco(dataset_path):
    """
    Convert  split-based COCO format to unified structure for training:
    
    From:
    dataset/train/_annotations.coco.json + images
    dataset/valid/_annotations.coco.json + images  
    dataset/test/_annotations.coco.json + images
    
    To:
    dataset_unified/images/train/ + images
    dataset_unified/images/val/ + images
    dataset_unified/images/test/ + images
    dataset_unified/annotations/train.json
    dataset_unified/annotations/val.json  
    dataset_unified/annotations/test.json
    """
    print("🔄 Converting to unified COCO structure for training...")
    
    unified_path = dataset_path + "_unified"
    if os.path.exists(unified_path):
        print(f"✅ Unified dataset already exists at: {unified_path}")
        return unified_path
    
    os.makedirs(unified_path, exist_ok=True)
    
    # Create unified structure
    images_dir = os.path.join(unified_path, 'images')
    annotations_dir = os.path.join(unified_path, 'annotations')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Map splits (valid -> val for consistency)
    split_mapping = {'train': 'train', 'valid': 'val', 'test': 'test'}
    
    total_images = 0
    total_annotations = 0
    categories = None
    
    for source_split, target_split in split_mapping.items():
        print(f"🔄 Processing {source_split} -> {target_split}...")
        
        source_path = os.path.join(dataset_path, source_split)
        if not os.path.exists(source_path):
            print(f"⚠️ Skipping {source_split} - not found")
            continue
        
        # Create target image directory
        target_img_dir = os.path.join(images_dir, target_split)
        os.makedirs(target_img_dir, exist_ok=True)
        
        # Read COCO annotations
        ann_file = os.path.join(source_path, '_annotations.coco.json')
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Extract categories (same for all splits)
        if categories is None:
            categories = coco_data['categories']
            print(f"📋 Found categories: {[cat['name'] for cat in categories]}")
        
        # Copy images to target directory
        source_images = [f for f in os.listdir(source_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in source_images:
            src_path = os.path.join(source_path, img_file)
            dst_path = os.path.join(target_img_dir, img_file)
            shutil.copy2(src_path, dst_path)
        
        # Save annotations to target
        target_ann_file = os.path.join(annotations_dir, f'{target_split}.json')
        with open(target_ann_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        # Statistics
        img_count = len(source_images)
        ann_count = len(coco_data['annotations'])
        total_images += img_count
        total_annotations += ann_count
        
        print(f"✅ {target_split}: {img_count} images, {ann_count} annotations")
    
    print(f"✅ Conversion complete!")
    print(f"📊 Total: {total_images} images, {total_annotations} annotations")
    print(f"📊 Classes: {len(categories)} road hazard categories")
    print(f"📁 Unified dataset: {unified_path}")
    
    return unified_path

class COCODataset(Dataset):
    """
    COCO format dataset loader for FocusNet road hazard detection
    
    Args:
        img_dir (str): Path to image directory (e.g., dataset_unified/images/train)
        ann_file (str): Path to COCO annotation file (e.g., dataset_unified/annotations/train.json)
        transforms: Image transforms to apply (from transforms_lowlight.py)
    """
    
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        
        # Load COCO annotations
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings for efficient access
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Group annotations by image_id
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        # Filter images that have annotations (for training)
        self.image_ids = [img_id for img_id in self.images.keys() 
                         if img_id in self.annotations and len(self.annotations[img_id]) > 0]
        
        print(f"Dataset loaded: {len(self.image_ids)} images with annotations")
        print(f"Total categories: {len(self.categories)}")
        print(f"Category names: {[cat['name'] for cat in self.categories.values()]}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            image = Image.new('RGB', (320, 320), color='black')
        
        # Get annotations for this image
        anns = self.annotations[img_id]
        
        # Convert annotations to tensors
        boxes = []
        labels = []
        areas = []
        iscrowds = []
        
        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2] format
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowds.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowds = torch.as_tensor(iscrowds, dtype=torch.uint8)
        
        # Normalize boxes to [0, 1] range (required by SSD)
        if len(boxes) > 0:
            img_w, img_h = image.size
            boxes[:, [0, 2]] /= img_w  # x coordinates
            boxes[:, [1, 3]] /= img_h  # y coordinates
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'areas': areas,
            'iscrowd': iscrowds,
            'image_id': torch.tensor([img_id])
        }
        
        # Apply transforms (from transforms_lowlight.py)
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, target
    
    def get_class_names(self):
        """Return list of class names for evaluation"""
        return [self.categories[cat_id]['name'] for cat_id in sorted(self.categories.keys())]
    
    def get_num_classes(self):
        """Return number of classes (including background)"""
        return len(self.categories) + 1  # +1 for background class

def collate_fn(batch):
    """Custom collate function for COCO dataset batching"""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)

def create_data_loaders(dataset_path_or_zip, batch_size=8, num_workers=2, img_size=320, 
                       extract_to="/content/dataset", force_extract=False):
    """
    Create train, validation, and test data loaders for FocusNet
    
    Args:
        dataset_path_or_zip (str): Path to dataset ZIP file in Google Drive OR extracted dataset directory
        batch_size (int): Batch size for training
        num_workers (int): Number of data loading workers  
        img_size (int): Input image size for FocusNet
        extract_to (str): Where to extract ZIP file (if provided)
        force_extract (bool): Re-extract even if exists
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
    """
    from transforms_lowlight import get_focusnet_train_transforms, get_focusnet_eval_transforms
    
    # Handle ZIP file extraction
    if dataset_path_or_zip.endswith('.zip'):
        print(f"📦 ZIP file provided: {dataset_path_or_zip}")
        dataset_path = extract_dataset_zip(dataset_path_or_zip, extract_to, force_extract)
    else:
        dataset_path = dataset_path_or_zip
        print(f"📁 Using dataset directory: {dataset_path}")
    
    # Verify  specific dataset structure
    if not verify_dataset_structure(dataset_path):
        raise ValueError("❌ Dataset structure validation failed!")
    
    # Convert to unified COCO structure for training
    unified_dataset_path = convert_to_unified_coco(dataset_path)
    
    # Define paths for unified structure
    images_base = os.path.join(unified_dataset_path, 'images')
    annotations_base = os.path.join(unified_dataset_path, 'annotations')
    
    train_img_dir = os.path.join(images_base, 'train')
    val_img_dir = os.path.join(images_base, 'val')
    test_img_dir = os.path.join(images_base, 'test')
    
    train_ann_file = os.path.join(annotations_base, 'train.json')
    val_ann_file = os.path.join(annotations_base, 'val.json')
    test_ann_file = os.path.join(annotations_base, 'test.json')
    
    # Get transforms optimized for low-light conditions
    train_transforms = get_focusnet_train_transforms(img_size)
    val_transforms = get_focusnet_eval_transforms(img_size)
    test_transforms = get_focusnet_eval_transforms(img_size)
    
    # Create datasets
    datasets = {}
    loaders = {}
    
    splits_info = [
        ('train', train_img_dir, train_ann_file, train_transforms, True),
        ('val', val_img_dir, val_ann_file, val_transforms, False),
        ('test', test_img_dir, test_ann_file, test_transforms, False)
    ]
    
    for split_name, img_dir, ann_file, transforms, shuffle in splits_info:
        if os.path.exists(img_dir) and os.path.exists(ann_file):
            try:
                dataset = COCODataset(img_dir, ann_file, transforms)
                datasets[split_name] = dataset
                
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=True,
                    drop_last=(split_name == 'train')  # Drop incomplete batches for training
                )
                loaders[split_name] = loader
                
                print(f"✅ {split_name} loader: {len(dataset)} samples, {len(loader)} batches")
                
            except Exception as e:
                print(f"❌ Failed to create {split_name} loader: {e}")
                datasets[split_name] = None
                loaders[split_name] = None
        else:
            print(f"⚠️ Skipped {split_name}: missing files")
            datasets[split_name] = None
            loaders[split_name] = None
    
    # Verify we have required datasets
    if not datasets.get('train'):
        raise ValueError("❌ Training dataset is required!")
    if not datasets.get('val'):
        raise ValueError("❌ Validation dataset is required!")
    
    print("✅ Data loaders created successfully!")
    print(f"📊 Training: {len(datasets['train'])} samples")
    print(f"📊 Validation: {len(datasets['val'])} samples") 
    print(f"📊 Test: {len(datasets['test']) if datasets['test'] else 0} samples")
    print(f"📊 Road hazard classes: {datasets['train'].get_num_classes() - 1}")  # -1 for background
    
    return (loaders['train'], loaders['val'], loaders['test'], 
            datasets['train'], datasets['val'], datasets['test'])

# Example usage for testing
if __name__ == "__main__":
    # Test with  dataset
    dataset_zip = "/content/drive/MyDrive/dataset.zip"  # Update this path
    
    try:
        train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_data_loaders(
            dataset_zip, 
            batch_size=4, 
            img_size=320,
            extract_to="/content/dataset"
        )
        
        print("✅ Dataset loading successful!")
        print(f"Classes: {train_ds.get_class_names()}")
        
        # Test batch loading
        for images, targets in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Targets: {len(targets)}")
            break
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")