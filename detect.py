"""
Main script for detecting AI-generated images.

Usage:
    python detect.py --image path/to/image.jpg
    python detect.py --image path/to/image.jpg --method ml_model
    python detect.py --image path/to/image.jpg --verbose
"""
import os
import sys
import argparse
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.detectors.ensemble_detector import EnsembleDetector
from src.utils.image_processing import load_image


def print_result(result, verbose=False):
    """
    Print detection result.

    Args:
        result: Detection result dictionary
        verbose: Whether to print detailed information
    """
    print("\n" + "="*60)
    print("AI-Generated Image Detection Result")
    print("="*60)

    if result['is_ai_generated']:
        print(f"🤖 VERDICT: AI-GENERATED")
    else:
        print(f"📷 VERDICT: REAL IMAGE")

    print(f"\nProbability of being AI-generated: {result['probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Threshold used: {result['threshold']:.2f}")
    print(f"Detection method: {result['method']}")

    if verbose and result['individual_predictions']:
        print("\n" + "-"*60)
        print("Individual Detector Results:")
        print("-"*60)

        for detector_name, prob in result['individual_predictions'].items():
            if prob is not None:
                print(f"  {detector_name.capitalize():15s}: {prob:.2%}")

    print("="*60 + "\n")


def detect_image(image_path, detector, method='weighted_average', threshold=0.5, verbose=False):
    """
    Detect if an image is AI-generated.

    Args:
        image_path: Path to the image
        detector: Ensemble detector instance
        method: Detection method to use
        threshold: Classification threshold
        verbose: Whether to print detailed information

    Returns:
        Detection result dictionary
    """
    # Load image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print("Running detection...")

    # Perform detection
    result = detector.predict(image, threshold=threshold, method=method)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Detect AI-generated images using ensemble detection methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect.py --image photo.jpg
  python detect.py --image photo.jpg --method ml_model --verbose
  python detect.py --image photo.jpg --threshold 0.7 --no-cnn

Detection Methods:
  weighted_average : Use weighted combination of all detectors (default)
  ml_model         : Use trained ML model on handcrafted features
  voting           : Use majority voting among detectors
        """
    )

    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image to analyze')
    parser.add_argument('--method', type=str, default='weighted_average',
                        choices=['weighted_average', 'ml_model', 'voting'],
                        help='Detection method to use')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (0.0-1.0)')
    parser.add_argument('--ml-model', type=str, default=None,
                        help='Path to trained ML model')
    parser.add_argument('--cnn-model', type=str, default=None,
                        help='Path to trained CNN model')
    parser.add_argument('--no-cnn', action='store_true',
                        help='Disable CNN detector (faster but less accurate)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed information')

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    # Initialize detector
    print("Initializing detector...")

    use_cnn = not args.no_cnn

    # Check for default model paths
    ml_model_path = args.ml_model
    if ml_model_path is None:
        default_ml_path = 'trained_models/ml_model_random_forest.pkl'
        if os.path.exists(default_ml_path):
            ml_model_path = default_ml_path

    cnn_model_path = args.cnn_model
    if cnn_model_path is None:
        default_cnn_path = 'trained_models/cnn_model_custom.pth'
        if os.path.exists(default_cnn_path):
            cnn_model_path = default_cnn_path

    detector = EnsembleDetector(
        ml_model_path=ml_model_path,
        cnn_model_path=cnn_model_path,
        use_cnn=use_cnn
    )

    if not use_cnn:
        print("Note: CNN detector disabled. Using only frequency and statistical analysis.")

    # Perform detection
    result = detect_image(
        args.image,
        detector,
        method=args.method,
        threshold=args.threshold,
        verbose=args.verbose
    )

    # Print result
    print_result(result, verbose=args.verbose)

    # Return appropriate exit code
    sys.exit(0 if result['is_ai_generated'] else 1)


if __name__ == '__main__':
    main()
