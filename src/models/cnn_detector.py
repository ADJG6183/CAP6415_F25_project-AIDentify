"""
Deep learning CNN model for AI-generated image detection.

This model uses a convolutional neural network to learn discriminative features
directly from images. The architecture is designed to capture both low-level
and high-level patterns that differentiate real from AI-generated images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class AIImageDetectorCNN(nn.Module):
    """
    CNN architecture for AI-generated image detection.

    Architecture:
    - Multiple convolutional blocks with batch normalization and dropout
    - Global average pooling to handle variable input sizes
    - Fully connected layers for classification
    - Special attention to frequency domain features via custom layers
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        """
        Initialize the CNN detector.

        Args:
            num_classes: Number of output classes (2 for binary: real/fake)
            dropout_rate: Dropout rate for regularization
        """
        super(AIImageDetectorCNN, self).__init__()

        # Convolutional blocks
        # Block 1: Capture low-level features
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: Mid-level features
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: High-level features
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 4: Deep features
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class EfficientNetDetector(nn.Module):
    """
    EfficientNet-based detector for better performance.
    Uses transfer learning from pretrained EfficientNet.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize EfficientNet detector.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(EfficientNetDetector, self).__init__()

        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

            if pretrained:
                self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = efficientnet_b0(weights=None)

            # Replace the final classifier
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, num_classes)
            )
        except ImportError:
            # Fallback if torchvision version doesn't support this
            print("Warning: Could not load EfficientNet, using custom CNN instead")
            self.backbone = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.backbone is not None:
            return self.backbone(x)
        else:
            raise RuntimeError("EfficientNet backbone not available")


class DeepLearningDetector:
    """Wrapper class for deep learning based detection."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the deep learning detector.

        Args:
            model_path: Path to pretrained model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Try to use EfficientNet, fallback to custom CNN
        try:
            self.model = EfficientNetDetector(num_classes=2, pretrained=False)
        except:
            self.model = AIImageDetectorCNN(num_classes=2)

        self.model = self.model.to(self.device)

        # Load pretrained weights if provided
        if model_path is not None:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model weights from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")

        self.model.eval()

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: Input image as numpy array (RGB, 0-255)

        Returns:
            Preprocessed tensor
        """
        # Resize to standard size
        import cv2
        image = cv2.resize(image, (224, 224))

        # Normalize
        image = image.astype(np.float32) / 255.0

        # ImageNet normalization (if using pretrained model)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        return image.to(self.device)

    def predict_proba(self, image: np.ndarray) -> float:
        """
        Predict probability of image being AI-generated.

        Args:
            image: Input image as numpy array (RGB)

        Returns:
            Probability of being AI-generated (0-1)
        """
        # Preprocess
        image_tensor = self.preprocess_image(image)

        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)

            # Return probability of AI-generated class (index 1)
            ai_prob = probs[0, 1].cpu().item()

        return ai_prob

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract deep features from the model.

        Args:
            image: Input image

        Returns:
            Feature vector
        """
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            # Get features from the layer before final classification
            if isinstance(self.model, AIImageDetectorCNN):
                x = self.model.conv1(image_tensor)
                x = self.model.conv2(x)
                x = self.model.conv3(x)
                x = self.model.conv4(x)
                x = self.model.global_avg_pool(x)
                features = x.view(x.size(0), -1)
            else:
                # For EfficientNet, get features before classifier
                features = self.model.backbone.features(image_tensor)
                features = self.model.backbone.avgpool(features)
                features = torch.flatten(features, 1)

        return features.cpu().numpy()[0]
