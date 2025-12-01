"""
Script to download datasets for AI image detection training.

Downloads:
- Real images from ImageNet-Mini (Kaggle)
- AI-generated images from DiffusionDB or other sources
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_kaggle_setup():
    """Check if Kaggle API is properly configured."""
    try:
        import kaggle
        return True
    except (ImportError, OSError) as e:
        print("❌ Kaggle API not properly configured!")
        print("\nTo setup Kaggle API:")
        print("1. pip install kaggle")
        print("2. Get API token from https://www.kaggle.com/settings")
        print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<You>\\.kaggle\\ (Windows)")
        print("4. chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)")
        return False


def download_imagenet_mini(output_dir):
    """
    Download ImageNet-Mini dataset from Kaggle.

    Args:
        output_dir: Directory to save images
    """
    print("\n" + "="*60)
    print("Downloading ImageNet-Mini (1000 real images)")
    print("="*60)

    if not check_kaggle_setup():
        return False

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Download dataset using Kaggle API
        dataset_name = "ifigotin/imagenetmini-1000"

        print(f"\nDownloading {dataset_name}...")
        print("This may take a few minutes...\n")

        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset_name,
            "-p", output_dir,
            "--unzip"
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        print("✅ Download complete!")

        # Move files to correct location if needed
        # Find where the images actually are
        for root, dirs, files in os.walk(output_dir):
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                print(f"\nFound {len(image_files)} images in {root}")

                # If images are in a subdirectory, move them up
                if root != output_dir:
                    print(f"Moving images to {output_dir}...")
                    import shutil
                    for img in image_files:
                        src = os.path.join(root, img)
                        dst = os.path.join(output_dir, img)
                        if not os.path.exists(dst):
                            shutil.move(src, dst)

        # Count final images
        final_count = len([f for f in os.listdir(output_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        print(f"\n✅ Successfully downloaded {final_count} real images to {output_dir}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a Kaggle account")
        print("2. Accept the dataset terms at: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000")
        print("3. Verify your kaggle.json is correctly placed")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def download_manual_instructions():
    """Print manual download instructions."""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nFor Real Images (ImageNet-Mini):")
    print("1. Go to: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000")
    print("2. Click 'Download' (requires Kaggle account)")
    print("3. Extract the zip file")
    print("4. Move all images to: data/real/")
    print("\nFor AI-Generated Images:")
    print("Option 1 - Download from Kaggle:")
    print("  - Search for AI-generated image datasets")
    print("  - Example: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
    print("\nOption 2 - Generate using online tools:")
    print("  - Stable Diffusion: https://huggingface.co/spaces/stabilityai/stable-diffusion")
    print("  - DALL-E Mini: https://www.craiyon.com/")
    print("  - Bing Image Creator: https://www.bing.com/images/create")
    print("\nOption 3 - Generate locally:")
    print("  python scripts/generate_ai_images.py --count 1000")
    print("\nAfter downloading, organize as:")
    print("  data/")
    print("  ├── real/           # Put real images here")
    print("  └── ai_generated/   # Put AI images here")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for AI image detection training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--real-only', action='store_true',
                        help='Download only real images (ImageNet-Mini)')
    parser.add_argument('--ai-only', action='store_true',
                        help='Download only AI-generated images')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Base output directory (default: data)')
    parser.add_argument('--manual', action='store_true',
                        help='Show manual download instructions')

    args = parser.parse_args()

    if args.manual:
        download_manual_instructions()
        return

    # Create base directories
    real_dir = os.path.join(args.output_dir, 'real')
    ai_dir = os.path.join(args.output_dir, 'ai_generated')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)

    success = True

    # Download real images
    if not args.ai_only:
        print("\n📥 Downloading real images...")
        if not download_imagenet_mini(real_dir):
            success = False
            print("\n💡 Tip: Use --manual flag to see manual download instructions")

    # Download AI images
    if not args.real_only:
        print("\n📥 AI-generated images...")
        print("\n⚠️  Automatic AI image download not yet implemented.")
        print("Please use one of these methods:")
        print("\n1. Generate locally:")
        print("   python scripts/generate_ai_images.py --count 1000 --output data/ai_generated/")
        print("\n2. Download from Kaggle/HuggingFace manually")
        print("\n3. Use --manual flag for detailed instructions:")
        print("   python scripts/download_dataset.py --manual")

    if success:
        print("\n" + "="*60)
        print("✅ Dataset download complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Verify dataset: python scripts/verify_dataset.py")
        print("2. Train models: python src/train.py --data_dir data --model_type both")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("⚠️  Download completed with errors")
        print("="*60)
        print("See instructions above or use --manual flag")
        print("="*60 + "\n")


if __name__ == '__main__':
    main()
