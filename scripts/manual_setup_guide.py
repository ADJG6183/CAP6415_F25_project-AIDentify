"""
Manual dataset setup guide - Use this if automated download fails.

This script provides instructions for manually downloading and setting up datasets.
"""
import os


def print_manual_instructions():
    """Print manual download instructions."""
    print("\n" + "="*70)
    print("  📥 MANUAL DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70 + "\n")

    print("If the automated setup isn't working, follow these manual steps:\n")

    # Step 1
    print("━" * 70)
    print("STEP 1: Download Real Images (ImageNet-Mini)")
    print("━" * 70)
    print("\n1. Go to: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000")
    print("2. Click the blue 'Download' button (you'll need a Kaggle account)")
    print("3. Save the downloaded ZIP file")
    print("4. Extract the ZIP file")
    print("5. Copy all images to: data/real/\n")

    print("Commands (after downloading):")
    print("  mkdir -p data/real")
    print("  unzip imagenetmini-1000.zip")
    print("  mv imagenetmini-1000/* data/real/")
    print("  # Or just manually copy all .jpg files to data/real/\n")

    # Step 2
    print("━" * 70)
    print("STEP 2: Download AI-Generated Images (DALL-E Dataset)")
    print("━" * 70)
    print("\n1. Go to: https://www.kaggle.com/datasets/superpotato9/dalle-recognition-dataset")
    print("2. Click the blue 'Download' button")
    print("3. Save the downloaded ZIP file")
    print("4. Extract the ZIP file")
    print("5. Copy all AI-generated images to: data/ai_generated/\n")

    print("Commands (after downloading):")
    print("  mkdir -p data/ai_generated")
    print("  unzip dalle-recognition-dataset.zip")
    print("  mv dalle-recognition-dataset/* data/ai_generated/")
    print("  # Or just manually copy all image files to data/ai_generated/\n")

    # Step 3
    print("━" * 70)
    print("STEP 3: Verify Your Setup")
    print("━" * 70)
    print("\nRun the verification script to check everything is ready:")
    print("  python scripts/verify_dataset.py\n")

    print("You should see:")
    print("  ✅ Real images:        1000+")
    print("  ✅ AI-generated:       1000+")
    print("  ✅ Dataset is ready for training!\n")

    # Step 4
    print("━" * 70)
    print("STEP 4: Train the Models")
    print("━" * 70)
    print("\nOnce verified, train your models:")
    print("  python src/train.py --data_dir data --model_type both --epochs 50\n")

    # Expected structure
    print("━" * 70)
    print("EXPECTED DIRECTORY STRUCTURE")
    print("━" * 70)
    print("""
data/
├── real/                  # ImageNet-Mini images here
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   └── ... (1000+ images)
└── ai_generated/          # DALL-E dataset images here
    ├── image_001.png
    ├── image_002.png
    └── ... (1000+ images)
""")

    # Troubleshooting
    print("━" * 70)
    print("TROUBLESHOOTING")
    print("━" * 70)
    print("\n❌ Issue: Can't download from Kaggle")
    print("   Solution: Make sure you have a Kaggle account and are logged in\n")

    print("❌ Issue: Download button is disabled")
    print("   Solution: Scroll down and accept the dataset terms/rules\n")

    print("❌ Issue: Wrong file structure after unzip")
    print("   Solution: Look inside extracted folders for the actual images")
    print("             Copy ALL .jpg/.png files to data/real/ or data/ai_generated/\n")

    print("❌ Issue: Not enough images")
    print("   Solution: Make sure you copied ALL images, not just a subfolder\n")

    print("━" * 70)
    print("✅ Once you see 1000+ images in each folder, you're ready to train!")
    print("="*70 + "\n")


if __name__ == '__main__':
    print_manual_instructions()

    # Check current status
    print("\n" + "="*70)
    print("  📊 CURRENT STATUS")
    print("="*70 + "\n")

    real_dir = "data/real"
    ai_dir = "data/ai_generated"

    def count_images(directory):
        if not os.path.exists(directory):
            return 0
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        return sum(1 for f in os.listdir(directory)
                  if os.path.splitext(f)[1].lower() in valid_extensions)

    real_count = count_images(real_dir)
    ai_count = count_images(ai_dir)

    print(f"Real images in data/real/:             {real_count}")
    print(f"AI images in data/ai_generated/:       {ai_count}")
    print()

    if real_count >= 1000 and ai_count >= 1000:
        print("✅ You have enough images! Ready to train!")
        print("\nNext step:")
        print("  python src/train.py --data_dir data --model_type both --epochs 50")
    elif real_count > 0 or ai_count > 0:
        print("⚠️  You have some images, but need more:")
        if real_count < 1000:
            print(f"   - Need {1000 - real_count} more real images")
        if ai_count < 1000:
            print(f"   - Need {1000 - ai_count} more AI images")
    else:
        print("⚠️  No images found yet. Follow the instructions above to download.")

    print("\n" + "="*70 + "\n")
