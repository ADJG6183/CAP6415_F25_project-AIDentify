"""
Training pipeline for AI-generated image detection models.

This script trains both the ML model (on handcrafted features) and
the CNN model (end-to-end learning).
"""
import os
import sys
import argparse
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.frequency_detector import FrequencyDomainDetector
from src.detectors.statistical_detector import StatisticalDetector
from src.models.cnn_detector import AIImageDetectorCNN, EfficientNetDetector
from src.utils.image_processing import load_image, normalize_image


class ImageDataset(Dataset):
    """Dataset for loading images and labels."""

    def __init__(self, image_paths, labels, transform=None, target_size=(224, 224)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        try:
            image = load_image(self.image_paths[idx], self.target_size)
        except:
            # Return a blank image if loading fails
            image = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)

        # Apply transforms if any
        if self.transform is not None:
            image = self.transform(image=image)['image']

        # Normalize
        image = normalize_image(image)

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        label = self.labels[idx]

        return image, label


def extract_features_from_dataset(image_paths, labels, freq_detector, stat_detector):
    """
    Extract features from all images in dataset.

    Args:
        image_paths: List of image paths
        labels: List of labels
        freq_detector: Frequency domain detector
        stat_detector: Statistical detector

    Returns:
        features, labels
    """
    all_features = []
    valid_labels = []

    print("Extracting features from images...")
    for i, img_path in enumerate(tqdm(image_paths)):
        try:
            # Load image
            image = load_image(img_path)

            # Extract features
            freq_features = freq_detector.extract_features(image)
            stat_features = stat_detector.extract_features(image)

            # Combine features
            combined_features = np.concatenate([freq_features, stat_features])
            all_features.append(combined_features)
            valid_labels.append(labels[i])

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    return np.array(all_features), np.array(valid_labels)


def train_ml_model(X_train, y_train, X_val, y_val, model_type='random_forest'):
    """
    Train machine learning model on handcrafted features.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_type: Type of ML model ('random_forest', 'gradient_boosting', 'svm')

    Returns:
        Trained model
    """
    print(f"\nTraining {model_type} model...")

    # Initialize model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=1
        )
    elif model_type == 'svm':
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    if hasattr(model, 'predict_proba'):
        val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        print(f"Validation AUC: {auc:.4f}")

    print(f"Precision: {precision_score(y_val, val_pred):.4f}")
    print(f"Recall: {recall_score(y_val, val_pred):.4f}")
    print(f"F1 Score: {f1_score(y_val, val_pred):.4f}")

    return model


def train_cnn_model(train_loader, val_loader, model_type='custom', num_epochs=50, device='cuda'):
    """
    Train CNN model.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model_type: Type of CNN ('custom' or 'efficientnet')
        num_epochs: Number of training epochs
        device: Device to train on

    Returns:
        Trained model
    """
    print(f"\nTraining {model_type} CNN model...")

    # Initialize model
    if model_type == 'efficientnet':
        try:
            model = EfficientNetDetector(num_classes=2, pretrained=True)
        except:
            print("EfficientNet not available, using custom CNN")
            model = AIImageDetectorCNN(num_classes=2)
    else:
        model = AIImageDetectorCNN(num_classes=2)

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_acc = 0.0
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  New best model! Val Acc: {best_val_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train AI-generated image detector')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing data')
    parser.add_argument('--model_type', type=str, default='ml', choices=['ml', 'cnn', 'both'],
                        help='Type of model to train')
    parser.add_argument('--ml_algorithm', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'svm'],
                        help='ML algorithm for feature-based model')
    parser.add_argument('--cnn_type', type=str, default='custom', choices=['custom', 'efficientnet'],
                        help='CNN architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs for CNN')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for CNN training')
    parser.add_argument('--output_dir', type=str, default='trained_models', help='Output directory for models')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    real_dir = os.path.join(args.data_dir, 'real')
    ai_dir = os.path.join(args.data_dir, 'ai_generated')

    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    ai_images = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    image_paths = real_images + ai_images
    labels = [0] * len(real_images) + [1] * len(ai_images)

    print(f"Loaded {len(real_images)} real images and {len(ai_images)} AI-generated images")

    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train ML model
    if args.model_type in ['ml', 'both']:
        freq_detector = FrequencyDomainDetector()
        stat_detector = StatisticalDetector()

        # Extract features
        X_train, y_train = extract_features_from_dataset(train_paths, train_labels, freq_detector, stat_detector)
        X_val, y_val = extract_features_from_dataset(val_paths, val_labels, freq_detector, stat_detector)

        # Train
        ml_model = train_ml_model(X_train, y_train, X_val, y_val, model_type=args.ml_algorithm)

        # Save model
        model_path = os.path.join(args.output_dir, f'ml_model_{args.ml_algorithm}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(ml_model, f)
        print(f"\nML model saved to {model_path}")

    # Train CNN model
    if args.model_type in ['cnn', 'both']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Create data loaders
        train_dataset = ImageDataset(train_paths, train_labels)
        val_dataset = ImageDataset(val_paths, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Train
        cnn_model = train_cnn_model(train_loader, val_loader, model_type=args.cnn_type,
                                     num_epochs=args.epochs, device=device)

        # Save model
        model_path = os.path.join(args.output_dir, f'cnn_model_{args.cnn_type}.pth')
        torch.save(cnn_model.state_dict(), model_path)
        print(f"\nCNN model saved to {model_path}")


if __name__ == '__main__':
    main()
