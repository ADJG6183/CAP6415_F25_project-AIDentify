"""
Split dataset into train and test sets.

Creates train/test split (80/20) while preserving class balance.
"""
import os
import shutil
import random
from pathlib import Path

def split_dataset(data_dir='data', train_ratio=0.8, seed=42):
    """
    Split data into train and test sets.

    Args:
        data_dir: Directory containing real/ and ai_generated/ folders
        train_ratio: Proportion of data for training (default 0.8 = 80%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Define directories
    real_dir = os.path.join(data_dir, 'real')
    ai_dir = os.path.join(data_dir, 'ai_generated')

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Create train/test directories
    os.makedirs(os.path.join(train_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'ai_generated'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'ai_generated'), exist_ok=True)

    # Process each class
    for class_name, source_dir in [('real', real_dir), ('ai_generated', ai_dir)]:
        print(f"\nProcessing {class_name} images...")

        # Get all image files
        image_files = [f for f in os.listdir(source_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Shuffle
        random.shuffle(image_files)

        # Split
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]

        print(f"  Total: {len(image_files)}")
        print(f"  Train: {len(train_files)}")
        print(f"  Test:  {len(test_files)}")

        # Copy to train
        train_dest = os.path.join(train_dir, class_name)
        for filename in train_files:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(train_dest, filename)
            shutil.copy2(src, dst)

        # Copy to test
        test_dest = os.path.join(test_dir, class_name)
        for filename in test_files:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(test_dest, filename)
            shutil.copy2(src, dst)

    print(f"\n✓ Dataset split complete!")
    print(f"  Train data: {train_dir}")
    print(f"  Test data:  {test_dir}")
    print(f"\nTo train: python src/train.py --data_dir {train_dir}")
    print(f"To evaluate: python src/evaluate.py --data_dir {test_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Split dataset into train/test')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory containing real/ and ai_generated/')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()
    split_dataset(args.data_dir, args.train_ratio, args.seed)
