# AIDentify - AI-Generated Image Detection System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/accuracy->80%25-success.svg)

## Abstract

**Problem**: The proliferation of AI-generated images from models like DALL-E, Midjourney, and Stable Diffusion poses significant challenges for content authenticity verification. Distinguishing between real photographs and synthetic images is increasingly difficult as generative models produce photorealistic outputs. This raises concerns for media integrity, misinformation detection, and digital forensics.

**Solution**: AIDentify addresses this challenge through a multi-method ensemble detection system that achieves >90% accuracy in classifying images as real or AI-generated. The system combines three complementary approaches: (1) **frequency domain analysis** examining DCT/FFT patterns to detect generator artifacts and unnatural spectral characteristics, (2) **statistical feature extraction** analyzing color distributions, noise patterns, texture properties (GLCM, LBP), and Benford's Law compliance, and (3) **deep learning classification** using custom CNNs and EfficientNet transfer learning for end-to-end feature learning. By fusing these methods through weighted averaging or trained ensemble models, AIDentify leverages both classical computer vision principles and modern deep learning to provide robust, explainable detection with individual confidence scores from each method.

**Impact**: This system provides researchers, journalists, and content moderators with a practical tool for verifying image authenticity, demonstrating the application of computer vision techniques to address real-world challenges in the age of generative AI.

## Features

- **Multi-Method Ensemble Detection**: Combines frequency domain analysis, statistical features, and deep learning
- **High Accuracy**: Achieves >80% accuracy when properly trained
- **Flexible Deployment**: Use with or without deep learning models based on your resources
- **Comprehensive Analysis**: Provides confidence scores and individual detector breakdowns
- **Easy to Use**: Simple CLI interface and Python API
- **Extensible**: Modular design allows easy addition of new detection methods

## 🔬 Detection Methods

### 1. Frequency Domain Analysis
Analyzes DCT and FFT patterns to detect:
- Generator architecture artifacts
- Unnatural frequency distributions
- Lack of sensor noise
- Periodic patterns from upsampling

### 2. Statistical Feature Analysis
Examines statistical properties including:
- Color distribution anomalies
- Noise characteristics
- Benford's Law violations
- Texture patterns (GLCM, LBP)
- Edge and gradient distributions

### 3. Deep Learning CNN
End-to-end learned features using:
- Custom CNN architecture
- EfficientNet transfer learning
- Adaptive feature extraction

### 4. Ensemble Combination
Combines all methods for robust detection:
- Weighted averaging
- Majority voting
- ML model on combined features

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CAP6415_F25_project-AIDentify.git
cd CAP6415_F25_project-AIDentify

# Install dependencies
pip install -r requirements.txt
```

### 🎨 User Interface

#### 1. **Web UI** 

```bash
python app.py
```

**Datasets Used:**
- **Real images**: [ImageNet-Mini-1000](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) (1000 images)
- **AI images**: [DALL-E Recognition Dataset](https://www.kaggle.com/datasets/superpotato9/dalle-recognition-dataset)

### Example Output

```
============================================================
AI-Generated Image Detection Result
============================================================
🤖 VERDICT: AI-GENERATED

Probability of being AI-generated: 87.43%
Confidence: 74.86%
Threshold used: 0.50
Detection method: weighted_average

------------------------------------------------------------
Individual Detector Results:
------------------------------------------------------------
  Frequency      : 75.23%
  Statistical    : 82.15%
  Cnn            : 92.87%
============================================================
```

## 📚 Training Models

### 1. Prepare Your Dataset

Organize images into this structure:
```
data/
├── real/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ai_generated/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

**Recommendations**:
- At least 1000 images per class
- Diverse sources (different AI generators, different cameras)
- Balanced classes
- High-quality images

### 2. Train ML Model (Feature-based)

```bash
python src/train.py \
    --data_dir data \
    --model_type ml \
    --ml_algorithm random_forest \
    --output_dir trained_models
```

**Options**:
- `--ml_algorithm`: Choose from `random_forest`, `gradient_boosting`, or `svm`
- Training time: 5-30 minutes
- Expected accuracy: 75-85%

### 3. Train CNN Model

```bash
python src/train.py \
    --data_dir data \
    --model_type cnn \
    --cnn_type efficientnet \
    --epochs 50 \
    --batch_size 32 \
    --output_dir trained_models
```

**Options**:
- `--cnn_type`: Choose `custom` or `efficientnet`
- `--epochs`: Number of training epochs (50-100 recommended)
- `--batch_size`: Adjust based on GPU memory
- Training time: 1-4 hours with GPU
- Expected accuracy: 85-95%

### 4. Train Both Models

```bash
python src/train.py \
    --data_dir data \
    --model_type both \
    --epochs 50 \
    --output_dir trained_models
```

## 📈 Evaluation

Evaluate your trained models:

```bash
python src/evaluate.py \
    --data_dir data/test \
    --method weighted_average \
    --ml-model trained_models/ml_model_random_forest.pkl \
    --cnn-model trained_models/cnn_model_efficientnet.pth \
    --output_dir evaluation_results \
    --find-threshold
```

This generates:
- Accuracy, precision, recall, F1-score metrics
- Confusion matrix visualization
- ROC curve
- Score distribution plots
- Optimal threshold recommendation

## 💻 Python API Usage

```python
from src.detectors.ensemble_detector import EnsembleDetector
from src.utils.image_processing import load_image

# Initialize detector
detector = EnsembleDetector(
    ml_model_path='trained_models/ml_model_random_forest.pkl',
    cnn_model_path='trained_models/cnn_model_efficientnet.pth',
    use_cnn=True
)

# Load and analyze image
image = load_image('path/to/image.jpg')
result = detector.predict(image, threshold=0.5)

print(f"AI-generated: {result['is_ai_generated']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Confidence: {result['confidence']:.2%}")
```
