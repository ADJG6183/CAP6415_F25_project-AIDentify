"""
Verify dataset structure and quality for AI image detection training.

Checks:
- Directory structure
- Image counts
- Image integrity
- Class balance
- File formats
"""
import os
import sys
from pathlib import Path
from PIL import Image
from collections import defaultdict


def verify_directory_structure(data_dir):
    """Verify the dataset has correct directory structure."""
    print("\n📁 Checking directory structure...")

    required_dirs = ['real', 'ai_generated']
    issues = []

    for dir_name in required_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.exists(dir_path):
            issues.append(f"Missing directory: {dir_path}")
        elif not os.path.isdir(dir_path):
            issues.append(f"Not a directory: {dir_path}")

    if issues:
        print("❌ Directory structure issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Directory structure correct")
        return True


def count_images(directory):
    """Count images in a directory."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    count = 0

    if not os.path.exists(directory):
        return 0

    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            count += 1

    return count


def verify_image_counts(data_dir, min_count=100):
    """Verify sufficient images in each category."""
    print(f"\n🖼️  Checking image counts (minimum: {min_count} per class)...")

    real_dir = os.path.join(data_dir, 'real')
    ai_dir = os.path.join(data_dir, 'ai_generated')

    real_count = count_images(real_dir)
    ai_count = count_images(ai_dir)

    print(f"   Real images: {real_count}")
    print(f"   AI-generated images: {ai_count}")

    issues = []

    if real_count < min_count:
        issues.append(f"Insufficient real images ({real_count} < {min_count})")

    if ai_count < min_count:
        issues.append(f"Insufficient AI images ({ai_count} < {min_count})")

    # Check balance
    if real_count > 0 and ai_count > 0:
        ratio = max(real_count, ai_count) / min(real_count, ai_count)
        if ratio > 2.0:
            issues.append(f"Imbalanced dataset (ratio: {ratio:.2f}:1)")
            print(f"   ⚠️  Dataset imbalance: {ratio:.2f}:1 (ideally < 2:1)")

    if issues:
        print("❌ Image count issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False, real_count, ai_count
    else:
        print("✅ Sufficient images in both classes")
        return True, real_count, ai_count


def verify_image_integrity(data_dir, sample_size=50):
    """Verify images can be loaded and are valid."""
    print(f"\n🔍 Checking image integrity (sampling {sample_size} images per class)...")

    real_dir = os.path.join(data_dir, 'real')
    ai_dir = os.path.join(data_dir, 'ai_generated')

    corrupted = []
    stats = defaultdict(int)

    for class_name, class_dir in [('real', real_dir), ('ai_generated', ai_dir)]:
        if not os.path.exists(class_dir):
            continue

        # Get list of images
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        images = [f for f in os.listdir(class_dir)
                 if os.path.splitext(f)[1].lower() in valid_extensions]

        # Sample images
        sample = images[:sample_size] if len(images) > sample_size else images

        for img_name in sample:
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    # Try to load the image
                    img.verify()

                # Reopen to get info (verify closes the file)
                with Image.open(img_path) as img:
                    stats[f'{class_name}_format_{img.format}'] += 1
                    stats[f'{class_name}_mode_{img.mode}'] += 1

            except Exception as e:
                corrupted.append((class_name, img_name, str(e)))

    if corrupted:
        print(f"❌ Found {len(corrupted)} corrupted images:")
        for class_name, img_name, error in corrupted[:10]:  # Show first 10
            print(f"   - {class_name}/{img_name}: {error}")
        if len(corrupted) > 10:
            print(f"   ... and {len(corrupted) - 10} more")
        return False
    else:
        print("✅ All sampled images are valid")

        # Print format statistics
        print("\n   Format distribution:")
        for key, value in sorted(stats.items()):
            if 'format' in key:
                print(f"      {key}: {value}")

        return True


def print_recommendations(real_count, ai_count):
    """Print recommendations based on dataset size."""
    print("\n💡 Recommendations:")

    total = real_count + ai_count

    if total < 1000:
        print("   📊 Dataset Size: Small")
        print("      - Expected accuracy: 70-80%")
        print("      - Recommendation: Add more images for better accuracy")
        print("      - Target: 1000+ images per class")
    elif total < 4000:
        print("   📊 Dataset Size: Good")
        print("      - Expected accuracy: 80-90%")
        print("      - Recommendation: Current size is good for >80% accuracy")
        print("      - Optional: Add more images for even better results")
    else:
        print("   📊 Dataset Size: Excellent")
        print("      - Expected accuracy: 90-97%")
        print("      - Recommendation: Dataset size is excellent!")

    # Balance check
    if real_count > 0 and ai_count > 0:
        ratio = max(real_count, ai_count) / min(real_count, ai_count)
        if ratio > 2.0:
            print(f"\n   ⚖️  Balance: Imbalanced ({ratio:.1f}:1)")
            print("      - Recommendation: Balance the classes for better results")
            larger = "real" if real_count > ai_count else "AI-generated"
            smaller = "AI-generated" if larger == "real" else "real"
            target = max(real_count, ai_count)
            print(f"      - Add {target - min(real_count, ai_count)} more {smaller} images")
        else:
            print(f"\n   ⚖️  Balance: Good ({ratio:.1f}:1)")

    print("\n   🎯 For >80% accuracy:")
    print("      1. Have at least 1000 images per class")
    print("      2. Keep classes balanced (ratio < 2:1)")
    print("      3. Use diverse image sources")
    print("      4. Train with both ML and CNN models")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Verify dataset for training')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to dataset directory (default: data)')
    parser.add_argument('--min-count', type=int, default=100,
                        help='Minimum images per class (default: 100)')
    parser.add_argument('--sample-size', type=int, default=50,
                        help='Number of images to verify per class (default: 50)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    print(f"\nDataset directory: {args.data_dir}")

    # Run checks
    checks_passed = 0
    total_checks = 3

    # Check 1: Directory structure
    if verify_directory_structure(args.data_dir):
        checks_passed += 1

    # Check 2: Image counts
    count_ok, real_count, ai_count = verify_image_counts(args.data_dir, args.min_count)
    if count_ok:
        checks_passed += 1

    # Check 3: Image integrity
    if verify_image_integrity(args.data_dir, args.sample_size):
        checks_passed += 1

    # Summary
    print("\n" + "="*60)
    print(f"VERIFICATION SUMMARY: {checks_passed}/{total_checks} checks passed")
    print("="*60)

    if checks_passed == total_checks:
        print("✅ Dataset is ready for training!")
        print_recommendations(real_count, ai_count)

        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Train models:")
        print("   python src/train.py --data_dir data --model_type both --epochs 50")
        print("\n2. Evaluate:")
        print("   python src/evaluate.py --data_dir data/test")
        print("\n3. Use the system:")
        print("   python app.py")
        print("\n" + "="*60)
    else:
        print("⚠️  Dataset has issues that should be fixed")
        print("\nCommon solutions:")
        print("- Download more images: python scripts/download_dataset.py")
        print("- Generate AI images: python scripts/generate_ai_images.py")
        print("- See manual: python scripts/download_dataset.py --manual")
        print("\n" + "="*60)

    # Return exit code
    sys.exit(0 if checks_passed == total_checks else 1)


if __name__ == '__main__':
    main()
