# Dataset Setup Guide

This guide shows you how to set up training data for the AI image detection system to achieve >80% accuracy.

## 📊 Dataset Requirements

For optimal performance (>80% accuracy), you need:
- **At least 1000 real images** (from cameras/photos)
- **At least 1000 AI-generated images** (from various generators)
- **Diverse sources** for both categories
- **Balanced dataset** (equal numbers in each class)

---

## 🖼️ Real Images Dataset

### Option 1: ImageNet-Mini (Recommended)

**Dataset:** [ImageNet-Mini-1000](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)

This dataset contains 1000 real images from ImageNet, perfect for training!

#### Download Instructions:

**Method A: Using Kaggle API (Recommended)**

1. Install Kaggle API:
```bash
pip install kaggle
```

2. Get Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json`

3. Setup credentials:
```bash
# Linux/Mac
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
# Move kaggle.json to C:\Users\<YourUsername>\.kaggle\
```

4. Download dataset:
```bash
# Run the provided script
python scripts/download_dataset.py --real-only
```

**Method B: Manual Download**

1. Go to: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
2. Click "Download" (requires Kaggle account)
3. Extract the downloaded zip file
4. Move images to `data/real/` folder:
```bash
unzip imagenetmini-1000.zip
mv imagenetmini-1000/* data/real/
```

### Option 2: Other Real Image Sources

Alternative sources for real images:

1. **COCO Dataset**: http://cocodataset.org/
2. **Open Images**: https://storage.googleapis.com/openimages/web/index.html
3. **Your own photos**: Use photos from cameras/smartphones
4. **Flickr**: Download Creative Commons licensed photos
5. **Unsplash**: High-quality free photos (https://unsplash.com/)

---

## 🤖 AI-Generated Images Dataset

You need AI-generated images from various sources for diversity.

### Recommended Sources:

#### 1. **DiffusionDB** (Large AI image dataset)
- **Link**: https://huggingface.co/datasets/poloclub/diffusiondb
- **Contains**: 2M+ Stable Diffusion generated images
- **Download**:
```bash
# Install huggingface-cli
pip install huggingface-hub

# Download subset (1000 images)
python scripts/download_dataset.py --ai-only --source diffusiondb
```

#### 2. **CIFAKE Dataset** (AI-generated images)
- **Link**: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- **Contains**: Mix of real and AI-generated images (you can use AI portion)

#### 3. **Generate Your Own** (Most diverse)

Use various AI image generators:

**Free Online Generators:**
- **Stable Diffusion**: https://huggingface.co/spaces/stabilityai/stable-diffusion
- **DALL-E Mini (Craiyon)**: https://www.craiyon.com/
- **Bing Image Creator**: https://www.bing.com/images/create
- **Leonardo.ai**: https://leonardo.ai/ (free tier)
- **Playground AI**: https://playgroundai.com/

**Using Stable Diffusion Locally:**
```bash
# Install stable-diffusion
pip install diffusers transformers accelerate

# Generate images
python scripts/generate_ai_images.py --count 1000 --output data/ai_generated/
```

#### 4. **Kaggle Datasets**

Search Kaggle for AI-generated image datasets:
- https://www.kaggle.com/datasets?search=ai+generated+images
- https://www.kaggle.com/datasets?search=synthetic+images

---

## 🗂️ Dataset Organization

Your final dataset structure should look like:

```
data/
├── real/                  # Real images (1000+)
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
├── ai_generated/          # AI-generated images (1000+)
│   ├── ai_0001.jpg
│   ├── ai_0002.jpg
│   └── ...
└── test/                  # Test set (optional but recommended)
    ├── real/
    │   └── ...
    └── ai_generated/
        └── ...
```

---

## 🚀 Quick Setup Script

I've created scripts to automate dataset download and preparation:

### Download ImageNet-Mini + Generate AI Images

```bash
# 1. Setup directories
python scripts/setup_dataset.py

# 2. Download real images (ImageNet-Mini)
python scripts/download_dataset.py --real-only

# 3. Generate AI images using Stable Diffusion
python scripts/generate_ai_images.py --count 1000

# 4. Verify dataset
python scripts/verify_dataset.py
```

### Or use the all-in-one script:

```bash
python scripts/prepare_full_dataset.py --real-count 1000 --ai-count 1000
```

---

## ✅ Dataset Verification

After setting up your dataset, verify it:

```bash
python scripts/verify_dataset.py
```

This will check:
- ✅ Correct directory structure
- ✅ Sufficient number of images in each class
- ✅ Image file integrity
- ✅ Class balance
- ✅ Image format consistency

---

## 📈 Training with Your Dataset

Once your dataset is ready:

### Step 1: Train ML Model (Fast)

```bash
python src/train.py \
    --data_dir data \
    --model_type ml \
    --ml_algorithm random_forest \
    --output_dir trained_models
```

**Expected**: 75-85% accuracy, ~10-20 minutes training

### Step 2: Train CNN Model (Best Accuracy)

```bash
python src/train.py \
    --data_dir data \
    --model_type cnn \
    --cnn_type efficientnet \
    --epochs 50 \
    --batch_size 32 \
    --output_dir trained_models
```

**Expected**: 85-95% accuracy, ~2-4 hours with GPU

### Step 3: Train Both (Recommended)

```bash
python src/train.py \
    --data_dir data \
    --model_type both \
    --epochs 50 \
    --output_dir trained_models
```

**Expected**: 90-97% ensemble accuracy

---

## 🎯 Tips for Best Results

### Dataset Quality

1. **Diverse Real Images**:
   - Multiple camera types (DSLR, smartphone, etc.)
   - Various lighting conditions
   - Different subjects (people, landscapes, objects)
   - Multiple resolutions

2. **Diverse AI Images**:
   - Multiple generators (DALL-E, Midjourney, Stable Diffusion, etc.)
   - Various styles (realistic, artistic, cartoon)
   - Different generation settings
   - Mix of quality levels

3. **Balanced Dataset**:
   - Equal numbers in each class (1000 real, 1000 AI)
   - If unbalanced, use class weights during training

4. **Clean Data**:
   - Remove corrupted images
   - Verify images are correctly labeled
   - Remove duplicates

### Training Tips

1. **Use Data Augmentation** (automatically done in training script):
   - Rotation, flipping, color jitter
   - Helps model generalize better

2. **Create Test Set**:
   - Set aside 20% of data for testing
   - Never train on test data
   - Test set should represent real-world distribution

3. **Monitor Training**:
   - Watch for overfitting (train accuracy >> validation accuracy)
   - Use early stopping if validation loss increases
   - Save best model based on validation accuracy

4. **Iterate**:
   - Start with small dataset to verify pipeline works
   - Gradually increase dataset size
   - Add more diverse images if accuracy plateaus

---

## 📊 Expected Results by Dataset Size

| Dataset Size | ML Accuracy | CNN Accuracy | Ensemble Accuracy | Training Time |
|--------------|-------------|--------------|-------------------|---------------|
| 500 + 500 | 70-75% | 75-80% | 80-85% | ~30 min |
| 1000 + 1000 | 75-85% | 85-90% | 85-92% | ~1 hour |
| 2000 + 2000 | 80-88% | 88-93% | 90-95% | ~2 hours |
| 5000 + 5000 | 85-90% | 90-95% | 92-97% | ~4-6 hours |

---

## 🔍 Testing Your Trained Model

After training, evaluate performance:

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
- Accuracy metrics
- Confusion matrix
- ROC curve
- Optimal threshold
- Classification report

---

## 🆘 Troubleshooting

### Issue: Not enough storage space

**Solution**:
- Use smaller dataset (500-750 per class still gives good results)
- Compress images to smaller size
- Use cloud storage for dataset

### Issue: Download scripts failing

**Solution**:
- Check internet connection
- Verify Kaggle API credentials
- Try manual download instead

### Issue: Low accuracy after training

**Solutions**:
- Increase dataset size
- Add more diverse images
- Train for more epochs
- Use ensemble method
- Check for data leakage (AI images in real folder)

### Issue: Training takes too long

**Solutions**:
- Reduce dataset size temporarily
- Use smaller batch size
- Train only ML model first (faster)
- Use GPU if available
- Reduce number of epochs

---

## 📚 Additional Resources

- **Dataset Search**: https://datasetsearch.research.google.com/
- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **Hugging Face Datasets**: https://huggingface.co/datasets
- **Papers with Code Datasets**: https://paperswithcode.com/datasets

---

## ✨ Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup dataset structure
mkdir -p data/{real,ai_generated,test/{real,ai_generated}}

# 3. Download real images (ImageNet-Mini)
# Follow Kaggle download instructions above

# 4. Get AI images (choose one method)
# - Download from Kaggle/HuggingFace
# - Generate using Stable Diffusion
# - Download from online generators

# 5. Verify dataset
python scripts/verify_dataset.py

# 6. Train models
python src/train.py --data_dir data --model_type both --epochs 50

# 7. Evaluate
python src/evaluate.py --data_dir data/test

# 8. Use the trained system!
python app.py
```

---

**With 1000 real images (ImageNet-Mini) + 1000 AI-generated images, you should achieve 85-95% accuracy!** 🎯
