"""
Complete dataset setup script - Downloads both real and AI-generated images.

This script automates the entire dataset preparation process:
1. Downloads ImageNet-Mini (1000 real images)
2. Downloads DALL-E Recognition Dataset (AI-generated images)
3. Organizes them into the correct structure
4. Verifies the dataset is ready for training

Usage:
    python scripts/setup_complete_dataset.py
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def check_kaggle_setup():
    """Check if Kaggle API is properly configured."""
    try:
        import kaggle
        print("✅ Kaggle API is configured")
        return True
    except (ImportError, OSError) as e:
        print("❌ Kaggle API not properly configured!")
        print("\n📝 Setup Instructions:")
        print("="*70)
        print("\n1. Install Kaggle API:")
        print("   pip install kaggle")
        print("\n2. Get your API token:")
        print("   - Go to: https://www.kaggle.com/settings")
        print("   - Scroll to 'API' section")
        print("   - Click 'Create New API Token'")
        print("   - This downloads kaggle.json")
        print("\n3. Install the token:")
        print("\n   Linux/Mac:")
        print("   mkdir -p ~/.kaggle")
        print("   mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("   chmod 600 ~/.kaggle/kaggle.json")
        print("\n   Windows:")
        print("   mkdir %USERPROFILE%\\.kaggle")
        print("   move %USERPROFILE%\\Downloads\\kaggle.json %USERPROFILE%\\.kaggle\\")
        print("\n4. Re-run this script")
        print("="*70 + "\n")
        return False


def download_kaggle_dataset(dataset_name, output_dir, description):
    """
    Download a dataset from Kaggle.

    Args:
        dataset_name: Kaggle dataset identifier (owner/dataset-name)
        output_dir: Directory to extract to
        description: Human-readable description for progress messages

    Returns:
        True if successful, False otherwise
    """
    print(f"\n📥 Downloading {description}...")
    print(f"   Dataset: {dataset_name}")
    print(f"   Output: {output_dir}")

    try:
        # Create temporary download directory
        temp_dir = os.path.join(output_dir, "_temp_download")
        os.makedirs(temp_dir, exist_ok=True)

        # Download and unzip
        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset_name,
            "-p", temp_dir,
            "--unzip"
        ]

        print(f"\n   Running: kaggle datasets download -d {dataset_name}")
        print("   This may take a few minutes...\n")

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Find and move image files
        moved_count = 0
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    src = os.path.join(root, filename)
                    dst = os.path.join(output_dir, filename)

                    # Avoid overwriting
                    if os.path.exists(dst):
                        base, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(dst):
                            dst = os.path.join(output_dir, f"{base}_{counter}{ext}")
                            counter += 1

                    shutil.move(src, dst)
                    moved_count += 1

        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"\n✅ Successfully downloaded {moved_count} images")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error downloading {description}")
        print(f"   Error: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Make sure you have a Kaggle account")
        print(f"   2. Accept dataset terms at: https://www.kaggle.com/datasets/{dataset_name}")
        print(f"   3. Verify kaggle.json is in the right place")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def count_images(directory):
    """Count image files in a directory."""
    if not os.path.exists(directory):
        return 0

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    count = 0

    for filename in os.listdir(directory):
        if os.path.splitext(filename)[1].lower() in valid_extensions:
            count += 1

    return count


def main():
    print_header("🚀 COMPLETE DATASET SETUP")

    print("This script will download and setup:")
    print("  1. ImageNet-Mini (1000 real images)")
    print("  2. DALL-E Recognition Dataset (AI-generated images)")
    print("\nTotal download size: ~500MB-1GB")
    print("Estimated time: 5-10 minutes\n")

    # Check Kaggle setup
    if not check_kaggle_setup():
        print("\n⚠️  Please setup Kaggle API first (see instructions above)")
        sys.exit(1)

    # Create directories
    print("📁 Creating directory structure...")
    base_dir = "data"
    real_dir = os.path.join(base_dir, "real")
    ai_dir = os.path.join(base_dir, "ai_generated")

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)
    print("✅ Directories created\n")

    # Dataset configurations
    datasets = [
        {
            "name": "ifigotin/imagenetmini-1000",
            "output": real_dir,
            "description": "ImageNet-Mini (Real Images)",
            "url": "https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000"
        },
        {
            "name": "superpotato9/dalle-recognition-dataset",
            "output": ai_dir,
            "description": "DALL-E Recognition Dataset (AI Images)",
            "url": "https://www.kaggle.com/datasets/superpotato9/dalle-recognition-dataset"
        }
    ]

    # Download datasets
    success_count = 0

    for i, dataset in enumerate(datasets, 1):
        print_header(f"Step {i}/2: {dataset['description']}")

        # Check if already has images
        existing_count = count_images(dataset['output'])
        if existing_count >= 500:
            print(f"✅ Directory already has {existing_count} images - skipping download")
            print(f"   (Delete {dataset['output']} to re-download)")
            success_count += 1
            continue

        if download_kaggle_dataset(
            dataset['name'],
            dataset['output'],
            dataset['description']
        ):
            success_count += 1
        else:
            print(f"\n⚠️  Failed to download {dataset['description']}")
            print(f"   You can download manually from: {dataset['url']}")

    # Summary
    print_header("📊 DATASET SUMMARY")

    real_count = count_images(real_dir)
    ai_count = count_images(ai_dir)
    total_count = real_count + ai_count

    print(f"Real images:        {real_count:4d} (in {real_dir})")
    print(f"AI-generated:       {ai_count:4d} (in {ai_dir})")
    print(f"Total:              {total_count:4d}")
    print()

    # Recommendations
    if real_count >= 1000 and ai_count >= 1000:
        print("✅ EXCELLENT! You have enough data for >80% accuracy")
        print("\n📈 Expected Performance:")
        print("   - ML Model:       75-85% accuracy")
        print("   - CNN Model:      85-95% accuracy")
        print("   - Ensemble:       90-95% accuracy ⭐")
    elif real_count >= 500 and ai_count >= 500:
        print("✅ GOOD! You have enough data for ~80% accuracy")
        print("\n📈 Expected Performance:")
        print("   - ML Model:       70-80% accuracy")
        print("   - CNN Model:      80-88% accuracy")
        print("   - Ensemble:       85-90% accuracy")
    else:
        print("⚠️  WARNING: Low image count")
        print(f"   Recommended: 1000+ images per class")
        print(f"   Current: {real_count} real, {ai_count} AI")

    # Check balance
    if real_count > 0 and ai_count > 0:
        ratio = max(real_count, ai_count) / min(real_count, ai_count)
        if ratio > 2.0:
            print(f"\n⚠️  Dataset is imbalanced (ratio: {ratio:.1f}:1)")
            print("   Recommendation: Balance classes for better results")
        else:
            print(f"\n✅ Dataset is balanced (ratio: {ratio:.1f}:1)")

    # Next steps
    if success_count == len(datasets):
        print_header("🎯 NEXT STEPS")

        print("Your dataset is ready! Here's what to do next:\n")

        print("1️⃣  Verify dataset (optional but recommended):")
        print("   python scripts/verify_dataset.py\n")

        print("2️⃣  Train the models:")
        print("   python src/train.py --data_dir data --model_type both --epochs 50\n")

        print("3️⃣  Evaluate performance:")
        print("   python src/evaluate.py --data_dir data\n")

        print("4️⃣  Start using the system:")
        print("   python app.py                    # Web UI")
        print("   python app_desktop.py            # Desktop UI")
        print("   python detect.py --image img.jpg # Command line\n")

        print("⏱️  Estimated training time:")
        print("   - ML Model only:  10-20 minutes")
        print("   - CNN Model:      1-3 hours (with GPU)")
        print("   - Both models:    1-3 hours (recommended for best accuracy)")

        print("\n" + "="*70)
        print("✅ Setup complete! You're ready to train! 🚀")
        print("="*70 + "\n")

        sys.exit(0)
    else:
        print_header("⚠️  SETUP INCOMPLETE")

        print(f"Downloaded: {success_count}/{len(datasets)} datasets\n")
        print("Some downloads failed. You can:")
        print("1. Fix the errors above and re-run this script")
        print("2. Download manually from Kaggle:")
        for dataset in datasets:
            print(f"   - {dataset['url']}")
        print("\n3. See manual download instructions:")
        print("   python scripts/download_dataset.py --manual")

        print("\n" + "="*70 + "\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
