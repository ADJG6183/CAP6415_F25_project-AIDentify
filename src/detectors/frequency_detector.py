"""
Frequency domain analysis for AI-generated image detection.

This detector analyzes the frequency domain characteristics of images.
AI-generated images often have distinct patterns in DCT/FFT domains due to:
1. Generator architecture artifacts
2. Upsampling operations
3. Lack of natural sensor noise
4. Periodic patterns in generated textures
"""
import cv2
import numpy as np
from scipy import fftpack
from scipy.stats import entropy
import pywt
from typing import Dict, Tuple


class FrequencyDomainDetector:
    """Detector based on frequency domain analysis."""

    def __init__(self):
        self.feature_names = [
            'dct_energy_ratio',
            'fft_high_freq_ratio',
            'fft_radial_mean_low',
            'fft_radial_mean_mid',
            'fft_radial_mean_high',
            'dct_block_variance',
            'fft_peak_frequency',
            'wavelet_energy_ratio',
            'spectral_entropy',
            'azimuthal_average_slope'
        ]

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract frequency domain features from an image.

        Args:
            image: Input image (RGB or grayscale)

        Returns:
            Feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        features = []

        # DCT-based features
        dct_features = self._extract_dct_features(gray)
        features.extend(dct_features)

        # FFT-based features
        fft_features = self._extract_fft_features(gray)
        features.extend(fft_features)

        # Wavelet-based features
        wavelet_features = self._extract_wavelet_features(gray)
        features.extend(wavelet_features)

        return np.array(features)

    def _extract_dct_features(self, gray: np.ndarray) -> list:
        """Extract DCT-based features."""
        # Compute DCT
        dct = cv2.dct(np.float32(gray))

        # Energy ratio between low and high frequencies
        h, w = dct.shape
        low_freq = dct[:h//4, :w//4]
        high_freq = dct[h//2:, w//2:]

        low_energy = np.sum(np.abs(low_freq))
        high_energy = np.sum(np.abs(high_freq))
        total_energy = np.sum(np.abs(dct))

        dct_energy_ratio = low_energy / (total_energy + 1e-10)

        # Block-wise DCT variance (AI images often have more uniform DCT patterns)
        block_size = 8
        block_variances = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = dct[i:i+block_size, j:j+block_size]
                block_variances.append(np.var(block))

        dct_block_variance = np.mean(block_variances)

        return [dct_energy_ratio, dct_block_variance]

    def _extract_fft_features(self, gray: np.ndarray) -> list:
        """Extract FFT-based features."""
        # Compute 2D FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2

        # Create radial frequency bins
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2)

        # High frequency ratio
        max_radius = min(h, w) // 2
        high_freq_mask = r > max_radius * 0.7
        high_freq_ratio = np.sum(magnitude_spectrum[high_freq_mask]) / (np.sum(magnitude_spectrum) + 1e-10)

        # Radial average in different frequency bands
        low_freq_mask = r < max_radius * 0.3
        mid_freq_mask = (r >= max_radius * 0.3) & (r < max_radius * 0.7)

        radial_mean_low = np.mean(magnitude_spectrum[low_freq_mask])
        radial_mean_mid = np.mean(magnitude_spectrum[mid_freq_mask])
        radial_mean_high = np.mean(magnitude_spectrum[high_freq_mask]) if np.any(high_freq_mask) else 0

        # Peak frequency location
        magnitude_spectrum[center_h-5:center_h+5, center_w-5:center_w+5] = 0  # Remove DC component
        peak_idx = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
        peak_frequency = np.sqrt((peak_idx[0] - center_h)**2 + (peak_idx[1] - center_w)**2)

        # Spectral entropy (AI images often have lower entropy in frequency domain)
        flat_spectrum = magnitude_spectrum.flatten()
        flat_spectrum = flat_spectrum / (np.sum(flat_spectrum) + 1e-10)
        spectral_entropy = entropy(flat_spectrum[flat_spectrum > 0])

        # Azimuthal average slope
        azimuthal_slope = self._compute_azimuthal_average_slope(magnitude_spectrum)

        return [
            high_freq_ratio,
            radial_mean_low,
            radial_mean_mid,
            radial_mean_high,
            peak_frequency,
            spectral_entropy,
            azimuthal_slope
        ]

    def _extract_wavelet_features(self, gray: np.ndarray) -> list:
        """Extract wavelet-based features."""
        # Perform 2D wavelet decomposition
        coeffs = pywt.dwt2(gray, 'db1')
        cA, (cH, cV, cD) = coeffs

        # Energy in different subbands
        energy_approx = np.sum(cA ** 2)
        energy_horizontal = np.sum(cH ** 2)
        energy_vertical = np.sum(cV ** 2)
        energy_diagonal = np.sum(cD ** 2)

        total_energy = energy_approx + energy_horizontal + energy_vertical + energy_diagonal

        # Ratio of detail to approximation energy
        detail_energy = energy_horizontal + energy_vertical + energy_diagonal
        wavelet_energy_ratio = detail_energy / (total_energy + 1e-10)

        return [wavelet_energy_ratio]

    def _compute_azimuthal_average_slope(self, magnitude_spectrum: np.ndarray) -> float:
        """Compute the slope of azimuthal average vs. frequency."""
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2).astype(int)

        # Compute azimuthal average
        max_r = min(h, w) // 2
        azimuthal_avg = []
        for radius in range(1, max_r, 5):
            mask = (r >= radius) & (r < radius + 5)
            if np.any(mask):
                azimuthal_avg.append(np.mean(magnitude_spectrum[mask]))

        if len(azimuthal_avg) < 2:
            return 0.0

        # Fit linear regression to log-log plot
        x_vals = np.arange(len(azimuthal_avg))
        y_vals = np.log(np.array(azimuthal_avg) + 1e-10)

        # Compute slope
        slope = np.polyfit(x_vals, y_vals, 1)[0]

        return slope

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
        # Simple heuristic based on observed patterns
        # AI-generated images typically have:
        # - Higher DCT energy ratio (more energy in low frequencies)
        # - Lower high frequency ratio in FFT
        # - Lower spectral entropy
        # - More uniform DCT block variance

        score = 0.0
        weights = []

        # DCT energy ratio (AI images tend to have higher ratio)
        if features[0] > 0.85:
            score += 0.15
        weights.append(0.15)

        # FFT high freq ratio (AI images tend to have lower ratio)
        if features[1] < 0.1:
            score += 0.15
        weights.append(0.15)

        # Spectral entropy (AI images tend to have lower entropy)
        if features[8] < 5.0:
            score += 0.2
        weights.append(0.2)

        # Wavelet energy ratio (AI images show different patterns)
        if features[7] < 0.3 or features[7] > 0.7:
            score += 0.1
        weights.append(0.1)

        return min(1.0, score)
