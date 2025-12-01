"""
Evaluation script for AI-generated image detection models.

This script evaluates the performance of the detection system on a test dataset.
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.ensemble_detector import EnsembleDetector
from utils.image_processing import load_image


def evaluate_on_dataset(detector, image_paths, labels, method='weighted_average', threshold=0.5):
    """
    Evaluate detector on a dataset.

    Args:
        detector: Ensemble detector instance
        image_paths: List of image paths
        labels: List of labels (0=real, 1=AI-generated)
        method: Detection method to use
        threshold: Classification threshold

    Returns:
        Dictionary of evaluation metrics
    """
    predictions = []
    probabilities = []

    print("Evaluating on dataset...")
    for img_path in tqdm(image_paths):
        try:
            # Load image
            image = load_image(img_path)

            # Get prediction
            prob = detector.predict_proba(image, method=method)
            pred = 1 if prob > threshold else 0

            probabilities.append(prob)
            predictions.append(pred)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Use random prediction for failed images
            probabilities.append(0.5)
            predictions.append(0)

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    labels = np.array(labels)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    # AUC-ROC
    try:
        auc = roc_auc_score(labels, probabilities)
    except:
        auc = 0.0

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Classification report
    report = classification_report(labels, predictions, target_names=['Real', 'AI-Generated'])

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'confusion_matrix': cm,
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': labels,
        'classification_report': report
    }

    return results


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Real', 'AI-Generated'],
                yticklabels=['Real', 'AI-Generated'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curve(labels, probabilities, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_score_distribution(labels, probabilities, save_path=None):
    """Plot score distribution for real vs AI-generated images."""
    real_scores = probabilities[labels == 0]
    ai_scores = probabilities[labels == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(real_scores, bins=50, alpha=0.5, label='Real Images', color='blue')
    plt.hist(ai_scores, bins=50, alpha=0.5, label='AI-Generated Images', color='red')
    plt.xlabel('Detection Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Score distribution saved to {save_path}")
    else:
        plt.show()

    plt.close()


def find_optimal_threshold(labels, probabilities):
    """Find optimal threshold based on F1 score."""
    thresholds = np.arange(0.0, 1.01, 0.01)
    f1_scores = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        f1 = f1_score(labels, predictions)
        f1_scores.append(f1)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    return optimal_threshold, optimal_f1


def main():
    parser = argparse.ArgumentParser(description='Evaluate AI-generated image detector')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing test data')
    parser.add_argument('--method', type=str, default='weighted_average',
                        choices=['weighted_average', 'ml_model', 'voting'],
                        help='Detection method to use')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--ml-model', type=str, default=None,
                        help='Path to trained ML model')
    parser.add_argument('--cnn-model', type=str, default=None,
                        help='Path to trained CNN model')
    parser.add_argument('--no-cnn', action='store_true',
                        help='Disable CNN detector')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory for plots and results')
    parser.add_argument('--find-threshold', action='store_true',
                        help='Find optimal threshold based on F1 score')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading test dataset...")
    real_dir = os.path.join(args.data_dir, 'real')
    ai_dir = os.path.join(args.data_dir, 'ai_generated')

    if not os.path.exists(real_dir) or not os.path.exists(ai_dir):
        print(f"Error: Test data directories not found")
        print(f"Expected: {real_dir} and {ai_dir}")
        sys.exit(1)

    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    ai_images = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    image_paths = real_images + ai_images
    labels = [0] * len(real_images) + [1] * len(ai_images)

    print(f"Loaded {len(real_images)} real images and {len(ai_images)} AI-generated images")

    # Initialize detector
    print("Initializing detector...")

    use_cnn = not args.no_cnn

    detector = EnsembleDetector(
        ml_model_path=args.ml_model,
        cnn_model_path=args.cnn_model,
        use_cnn=use_cnn
    )

    # Evaluate
    results = evaluate_on_dataset(detector, image_paths, labels,
                                   method=args.method, threshold=args.threshold)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print(f"AUC-ROC:   {results['auc_roc']:.4f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("\n" + results['classification_report'])
    print("="*60 + "\n")

    # Find optimal threshold
    if args.find_threshold:
        optimal_threshold, optimal_f1 = find_optimal_threshold(
            results['labels'], results['probabilities']
        )
        print(f"Optimal threshold (based on F1): {optimal_threshold:.3f}")
        print(f"F1 score at optimal threshold: {optimal_f1:.4f}\n")

    # Generate plots
    print("Generating plots...")

    plot_confusion_matrix(
        results['confusion_matrix'],
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )

    plot_roc_curve(
        results['labels'],
        results['probabilities'],
        save_path=os.path.join(args.output_dir, 'roc_curve.png')
    )

    plot_score_distribution(
        results['labels'],
        results['probabilities'],
        save_path=os.path.join(args.output_dir, 'score_distribution.png')
    )

    print(f"\nResults saved to {args.output_dir}/")

    # Check if accuracy meets requirement
    if results['accuracy'] >= 0.80:
        print(f"\n✓ SUCCESS: Achieved {results['accuracy']*100:.2f}% accuracy (>80% requirement met!)")
    else:
        print(f"\n⚠ WARNING: Accuracy is {results['accuracy']*100:.2f}% (below 80% target)")
        print("  Consider:")
        print("  - Training on more diverse dataset")
        print("  - Using ensemble with CNN detector")
        print("  - Adjusting detection threshold")


if __name__ == '__main__':
    main()
