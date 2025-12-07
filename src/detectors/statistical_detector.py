"""
Statistical feature analysis for AI-generated image detection.

This detector analyzes statistical properties of images including:
1. Color distribution patterns
2. Noise characteristics (real images have sensor noise, AI images have different noise)
3. Local statistics and texture patterns
4. Edge and gradient distributions
5. Benford's Law violations (AI images often violate natural digit distributions)
"""
import cv2
import numpy as np
from scipy import stats
from scipy.ndimage import generic_filter
from skimage import feature, filters
from typing import Dict, List


class StatisticalDetector:
    """Detector based on statistical feature analysis."""

    def __init__(self):
        self.feature_names = [
            'color_variance_r',
            'color_variance_g',
            'color_variance_b',
            'color_skewness_mean',
            'color_kurtosis_mean',
            'noise_std',
            'noise_uniformity',
            'edge_density',
            'gradient_magnitude_mean',
            'gradient_magnitude_std',
            'benford_deviation',
            'local_variance_mean',
            'local_variance_std',
            'saturation_mean',
            'saturation_std',
            'glcm_contrast',
            'glcm_homogeneity',
            'glcm_energy',
            'lbp_uniformity'
        ]

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from an image.

        Args:
            image: Input image (RGB)

        Returns:
            Feature vector
        """
        features = []

        # Color distribution features
        color_features = self._extract_color_features(image)
        features.extend(color_features)

        # Noise characteristics
        noise_features = self._extract_noise_features(image)
        features.extend(noise_features)

        # Edge and gradient features
        edge_features = self._extract_edge_features(image)
        features.extend(edge_features)

        # Benford's Law analysis
        benford_feature = self._check_benford_law(image)
        features.append(benford_feature)

        # Local statistics
        local_features = self._extract_local_statistics(image)
        features.extend(local_features)

        # Texture features
        texture_features = self._extract_texture_features(image)
        features.extend(texture_features)

        # Ensure consistent dtype (float64) for all features
        return np.array(features, dtype=np.float64)

    def _extract_color_features(self, image: np.ndarray) -> List[float]:
        """Extract color distribution features."""
        if len(image.shape) == 2:
            # Grayscale image
            gray = image
            r = g = b = gray
        else:
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Variance in each channel
        var_r = np.var(r)
        var_g = np.var(g)
        var_b = np.var(b)

        # Skewness and kurtosis (AI images often have different distributions)
        skewness = np.mean([stats.skew(r.flatten()), stats.skew(g.flatten()), stats.skew(b.flatten())])
        kurtosis = np.mean([stats.kurtosis(r.flatten()), stats.kurtosis(g.flatten()), stats.kurtosis(b.flatten())])

        # Saturation statistics (HSV color space)
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            sat_mean = np.mean(saturation)
            sat_std = np.std(saturation)
        else:
            sat_mean = 0.0
            sat_std = 0.0

        return [var_r, var_g, var_b, skewness, kurtosis, sat_mean, sat_std]

    def _extract_noise_features(self, image: np.ndarray) -> List[float]:
        """
        Extract noise characteristics.
        Real images have natural sensor noise, AI images have different noise patterns.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Estimate noise using high-pass filter
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(np.float32) - blurred.astype(np.float32)

        # Noise standard deviation
        noise_std = np.std(noise)

        # Noise uniformity (AI images often have more uniform noise)
        # Divide image into blocks and compute noise variance in each
        block_size = 32
        h, w = gray.shape
        block_noises = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block_noise = noise[i:i+block_size, j:j+block_size]
                block_noises.append(np.std(block_noise))

        # Uniformity measured as inverse of variance of block noises
        noise_uniformity = 1.0 / (np.var(block_noises) + 1e-10)

        return [noise_std, noise_uniformity]

    def _extract_edge_features(self, image: np.ndarray) -> List[float]:
        """Extract edge and gradient features."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        grad_mean = np.mean(grad_magnitude)
        grad_std = np.std(grad_magnitude)

        return [edge_density, grad_mean, grad_std]

    def _check_benford_law(self, image: np.ndarray) -> float:
        """
        Check adherence to Benford's Law.
        Natural images tend to follow Benford's Law for leading digits.
        AI-generated images often violate this.
        """
        # Convert to grayscale and get non-zero pixel values
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Get first digits of non-zero pixel values
        non_zero_pixels = gray[gray > 0].flatten()

        if len(non_zero_pixels) == 0:
            return 0.0

        # Extract first digit
        first_digits = []
        for pixel in non_zero_pixels:
            digit = int(str(int(pixel))[0])
            if digit > 0:
                first_digits.append(digit)

        if len(first_digits) == 0:
            return 0.0

        # Benford's Law expected distribution
        benford_expected = np.array([np.log10(1 + 1/d) for d in range(1, 10)])

        # Observed distribution
        digit_counts = np.zeros(9)
        for digit in first_digits:
            digit_counts[digit - 1] += 1

        observed = digit_counts / len(first_digits)

        # Compute KL divergence from Benford's Law
        observed = np.maximum(observed, 1e-10)  # Avoid log(0)
        benford_deviation = np.sum(observed * np.log(observed / benford_expected))

        return benford_deviation

    def _extract_local_statistics(self, image: np.ndarray) -> List[float]:
        """Extract local statistical features."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Compute local variance using a sliding window
        def local_variance(values):
            return np.var(values)

        local_var = generic_filter(gray.astype(np.float32), local_variance, size=5)

        local_var_mean = np.mean(local_var)
        local_var_std = np.std(local_var)

        return [local_var_mean, local_var_std]

    def _extract_texture_features(self, image: np.ndarray) -> List[float]:
        """Extract texture features using GLCM and LBP."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Resize for faster computation if image is too large
        if gray.shape[0] > 512 or gray.shape[1] > 512:
            gray = cv2.resize(gray, (512, 512))

        # GLCM features
        from skimage.feature import graycomatrix, graycoprops

        # Compute GLCM
        gray_uint8 = gray.astype(np.uint8)
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        glcm = graycomatrix(gray_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        # Extract GLCM properties
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()

        # LBP (Local Binary Patterns)
        radius = 1
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')

        # LBP histogram uniformity
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-10)
        lbp_uniformity = np.sum(lbp_hist ** 2)  # Measure of uniformity

        return [contrast, homogeneity, energy, lbp_uniformity]

    def predict_proba(self, features: np.ndarray) -> float:
        """
        Predict probability of image being AI-generated based on features.
        This is a simple heuristic-based prediction.
        For better results, train an ML model on these features.

        Args:
            features: Feature vector

        Returns:
            Probability of being AI-generated (0-1)
        """
        score = 0.0

        # Noise characteristics (AI images often have more uniform noise)
        noise_uniformity = features[6]
        if noise_uniformity > 0.01:  # High uniformity suggests AI
            score += 0.2

        # Edge density (AI images sometimes have unusual edge patterns)
        edge_density = features[7]
        if edge_density < 0.05 or edge_density > 0.3:
            score += 0.1

        # Benford's Law deviation (higher deviation suggests AI)
        benford_dev = features[10]
        if benford_dev > 0.1:
            score += 0.15

        # Texture uniformity (AI images often have more uniform textures)
        lbp_uniformity = features[18]
        if lbp_uniformity > 0.15:
            score += 0.15

        return min(1.0, score)
