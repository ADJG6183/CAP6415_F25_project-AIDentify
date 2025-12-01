"""
Example usage script for AI-generated image detection.

This demonstrates how to use the detection system programmatically.
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.detectors.frequency_detector import FrequencyDomainDetector
from src.detectors.statistical_detector import StatisticalDetector
from src.detectors.ensemble_detector import EnsembleDetector
from src.utils.image_processing import load_image


def example_1_basic_detection():
    """Example 1: Basic detection without training."""
    print("\n" + "="*60)
    print("Example 1: Basic Detection (No Training Required)")
    print("="*60)

    # Create a synthetic test image
    # In practice, you would load a real image
    test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # Initialize detector (fast mode, no CNN)
    detector = EnsembleDetector(use_cnn=False)

    # Predict
    result = detector.predict(test_image)

    print(f"\nResult: {'AI-Generated' if result['is_ai_generated'] else 'Real Image'}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nIndividual predictions:")
    for method, prob in result['individual_predictions'].items():
        if prob is not None:
            print(f"  {method}: {prob:.2%}")


def example_2_individual_detectors():
    """Example 2: Using individual detectors."""
    print("\n" + "="*60)
    print("Example 2: Individual Detectors")
    print("="*60)

    # Create test image
    test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # Frequency domain detector
    freq_detector = FrequencyDomainDetector()
    freq_features = freq_detector.extract_features(test_image)
    freq_prob = freq_detector.predict_proba(freq_features)

    print("\nFrequency Domain Detector:")
    print(f"  Features: {len(freq_features)}")
    print(f"  AI probability: {freq_prob:.2%}")
    print(f"  Top features:")
    for i, (name, value) in enumerate(zip(freq_detector.feature_names[:5], freq_features[:5])):
        print(f"    {name}: {value:.4f}")

    # Statistical detector
    stat_detector = StatisticalDetector()
    stat_features = stat_detector.extract_features(test_image)
    stat_prob = stat_detector.predict_proba(stat_features)

    print("\nStatistical Detector:")
    print(f"  Features: {len(stat_features)}")
    print(f"  AI probability: {stat_prob:.2%}")
    print(f"  Top features:")
    for i, (name, value) in enumerate(zip(stat_detector.feature_names[:5], stat_features[:5])):
        print(f"    {name}: {value:.4f}")


def example_3_with_trained_models():
    """Example 3: Using trained models (if available)."""
    print("\n" + "="*60)
    print("Example 3: With Trained Models")
    print("="*60)

    ml_model_path = 'trained_models/ml_model_random_forest.pkl'
    cnn_model_path = 'trained_models/cnn_model_custom.pth'

    if os.path.exists(ml_model_path):
        print(f"\nFound ML model: {ml_model_path}")

        # Create test image
        test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        # Initialize with ML model
        detector = EnsembleDetector(ml_model_path=ml_model_path, use_cnn=False)

        # Predict using ML model
        result = detector.predict(test_image, method='ml_model')

        print(f"Result: {'AI-Generated' if result['is_ai_generated'] else 'Real Image'}")
        print(f"Probability: {result['probability']:.2%}")

        # Get feature importance
        importance = detector.get_feature_importance()
        if importance is not None:
            print(f"\nTop 5 most important features:")
            top_indices = np.argsort(importance)[::-1][:5]
            freq_detector = FrequencyDomainDetector()
            stat_detector = StatisticalDetector()
            all_names = freq_detector.feature_names + stat_detector.feature_names
            for idx in top_indices:
                print(f"  {all_names[idx]}: {importance[idx]:.4f}")
    else:
        print(f"\nNo trained models found at {ml_model_path}")
        print("Run training first: python src/train.py --data_dir data --model_type ml")


def example_4_batch_processing():
    """Example 4: Batch processing multiple images."""
    print("\n" + "="*60)
    print("Example 4: Batch Processing")
    print("="*60)

    # Simulate multiple images
    images = {
        'image1.jpg': np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
        'image2.jpg': np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
        'image3.jpg': np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
    }

    # Initialize detector
    detector = EnsembleDetector(use_cnn=False)

    print("\nProcessing images...")
    results = []
    for name, image in images.items():
        result = detector.predict(image)
        results.append({
            'name': name,
            'is_ai': result['is_ai_generated'],
            'probability': result['probability']
        })

    # Display results
    print("\nResults:")
    print(f"{'Image':<15} {'Verdict':<15} {'Probability':<12}")
    print("-" * 45)
    for r in results:
        verdict = "AI-Generated" if r['is_ai'] else "Real"
        print(f"{r['name']:<15} {verdict:<15} {r['probability']:<12.2%}")


def example_5_custom_ensemble():
    """Example 5: Custom ensemble weights."""
    print("\n" + "="*60)
    print("Example 5: Custom Ensemble Weights")
    print("="*60)

    test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # Try different weight configurations
    weight_configs = [
        {'frequency': 0.5, 'statistical': 0.5, 'cnn': 0.0},
        {'frequency': 0.7, 'statistical': 0.3, 'cnn': 0.0},
        {'frequency': 0.3, 'statistical': 0.7, 'cnn': 0.0},
    ]

    print("\nTesting different ensemble weights:")
    for i, weights in enumerate(weight_configs, 1):
        detector = EnsembleDetector(weights=weights, use_cnn=False)
        result = detector.predict(test_image)

        print(f"\nConfiguration {i}:")
        print(f"  Weights: freq={weights['frequency']:.1f}, stat={weights['statistical']:.1f}")
        print(f"  Probability: {result['probability']:.2%}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("AI-Generated Image Detection - Example Usage")
    print("="*60)

    try:
        example_1_basic_detection()
    except Exception as e:
        print(f"Error in Example 1: {e}")

    try:
        example_2_individual_detectors()
    except Exception as e:
        print(f"Error in Example 2: {e}")

    try:
        example_3_with_trained_models()
    except Exception as e:
        print(f"Error in Example 3: {e}")

    try:
        example_4_batch_processing()
    except Exception as e:
        print(f"Error in Example 4: {e}")

    try:
        example_5_custom_ensemble()
    except Exception as e:
        print(f"Error in Example 5: {e}")

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare training data in data/real and data/ai_generated")
    print("2. Train models: python src/train.py --data_dir data --model_type both")
    print("3. Evaluate: python src/evaluate.py --data_dir data/test")
    print("4. Detect: python detect.py --image your_image.jpg")
    print("\nSee README.md and IMPLEMENTATION_GUIDE.md for more details!")


if __name__ == '__main__':
    main()
