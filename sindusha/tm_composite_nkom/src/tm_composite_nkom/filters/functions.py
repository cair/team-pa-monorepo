from abc import ABC, abstractmethod

import cv2
import numpy as np
from scipy.ndimage import rotate
from skimage.feature import canny
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks

from tmu.composite.components.base import TMComponent

class ImagePreprocessor(ABC):
    @abstractmethod
    def process(self, images):
        pass

    def _ensure_3d(self, images):
        if images.ndim == 4 and images.shape[-1] == 1:
            return np.squeeze(images, axis=-1)
        elif images.ndim == 3:
            return images
        else:
            raise ValueError("Input must be a 3D array with shape (batch, width, height) or a 4D array with shape (batch, width, height, 1)")

    def __str__(self):
        return self.__class__.__name__

class CompositePreprocessor(ImagePreprocessor):
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def process(self, images):
        for preprocessor in self.preprocessors:
            images = preprocessor.process(images)
        return images

    def __str__(self):
        return f"{self.__class__.__name__}_{'_'.join(str(p) for p in self.preprocessors)}"

class FlexibleComponent(TMComponent):
    def __init__(self, model_cls, model_config, preprocessor, **kwargs):
        super().__init__(model_cls=model_cls, model_config=model_config, **kwargs)
        self.preprocessor = preprocessor

    def preprocess(self, data: dict):
        super().preprocess(data=data)
        X = data["X"]
        Y = data["Y"]
        X_processed = self.preprocessor.process(X)

        # ensure that the processed images are in the correct shape
        # batch, height, width, channels
        if len(X_processed.shape) == 3:
            X_processed = X_processed[..., np.newaxis]

        return dict(
            X=X_processed,
            Y=Y,
        )

    def __str__(self):
        return f"{self.preprocessor}"  # {self.__class__.__name__}_

# We're testing this to see if orienting spectral-temporal patterns differently
# improves feature detection. Tsetlin Machines might benefit from seeing these
# patterns from various angles, potentially uncovering discriminative features
# that are not apparent in the original orientation.
class RotationPreprocessor(ImagePreprocessor):
    def __init__(self, rotation_angle):
        self.rotation_angle = rotation_angle

    def process(self, images):
        if self.rotation_angle == 0:
            return images
        rotated_images = []
        for image in images:
            rotated = rotate(image, self.rotation_angle, reshape=False, order=1, mode='constant', cval=0)
            threshold = threshold_otsu(rotated)
            binary = (rotated > threshold).astype(np.uint8)
            rotated_images.append(binary)
        return np.array(rotated_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.rotation_angle}"

# This preprocessor is being evaluated to determine if a binary representation
# of spectrograms enhances the Tsetlin Machine's ability to distinguish between
# significant spectral components and background noise, potentially improving
# classification accuracy.
class OtsuThresholdingPreprocessor(ImagePreprocessor):
    def process(self, images):
        binary_images = []
        for image in images:
            threshold = threshold_otsu(image)
            binary = (image > threshold).astype(np.uint8)
            binary_images.append(binary)
        return np.array(binary_images)

# We're testing this to assess whether local adaptive thresholding can better
# capture varying intensity patterns across different frequency bands and time
# periods in spectrograms, potentially providing more nuanced input for the
# Tsetlin Machine.
class AdaptiveThresholdingPreprocessor(ImagePreprocessor):
    def __init__(self, block_size=11, C=2):
        self.block_size = block_size
        self.C = C

    def process(self, images):
        processed_images = []
        for image in images:
            binary = cv2.adaptiveThreshold(image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, self.block_size, self.C)
            processed_images.append(binary)
        return np.array(processed_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.block_size}_{self.C}"

# This is being evaluated to see if emphasizing spectro-temporal boundaries
# in the input helps the Tsetlin Machine identify key transitions in the audio
# signal, which could be crucial for distinguishing between different classes.
class CannyEdgePreprocessor(ImagePreprocessor):
    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def process(self, images):
        processed_images = []
        for image in images:
            edges = cv2.Canny(image, self.low_threshold, self.high_threshold)
            binary = (edges > 0).astype(np.uint8)
            processed_images.append(binary)
        return np.array(processed_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.low_threshold}_{self.high_threshold}"

# We're testing this to determine if enhancing the contrast in spectrograms
# allows the Tsetlin Machine to better recognize subtle patterns that might
# be significant for classification but are not prominent in the original image.
class HistogramEqualizationPreprocessor(ImagePreprocessor):
    def process(self, images):
        processed_images = []
        for image in images:
            equalized = cv2.equalizeHist(image)
            threshold = threshold_otsu(equalized)
            binary = (equalized > threshold).astype(np.uint8)
            processed_images.append(binary)
        return np.array(processed_images)

    def __str__(self):
        return f"{self.__class__.__name__}"

# This preprocessor is being evaluated to see if cleaning up noise and filling
# small gaps in the spectrogram creates more coherent regions, potentially
# making it easier for the Tsetlin Machine to identify consistent patterns.
class MorphologicalPreprocessor(ImagePreprocessor):
    def __init__(self, operation='open', kernel_size=5):
        self.operation = operation
        self.kernel_size = kernel_size
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def process(self, images):
        processed_images = []
        for image in images:
            threshold = threshold_otsu(image)
            binary = (image > threshold).astype(np.uint8)
            if self.operation == 'open':
                processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
            elif self.operation == 'close':
                processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
            processed_images.append(processed)
        return np.array(processed_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.operation}_{self.kernel_size}"

# We're testing this to assess whether smoothing out short-term variations
# helps the Tsetlin Machine focus on more stable, longer-term patterns in
# the audio, which could be more reliable for classification.
class GaussianBlurPreprocessor(ImagePreprocessor):
    def __init__(self, kernel_size=(5, 5), sigma=0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def process(self, images):
        blurred_images = []
        for image in images:
            blurred = cv2.GaussianBlur(image, self.kernel_size, self.sigma)
            threshold = threshold_otsu(blurred)
            binary = (blurred > threshold).astype(np.uint8)
            blurred_images.append(binary)
        return np.array(blurred_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.kernel_size}_{self.sigma}"

# This is being evaluated to see if highlighting rapid changes in frequency
# content over time provides the Tsetlin Machine with valuable information
# about onsets, offsets, and transitions in the audio signal.
class SobelEdgePreprocessor(ImagePreprocessor):
    def __init__(self, ksize=3):
        self.ksize = ksize

    def process(self, images):
        edge_images = []
        for image in images:
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.ksize)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.ksize)
            edge = np.sqrt(sobelx**2 + sobely**2)
            threshold = threshold_otsu(edge)
            binary = (edge > threshold).astype(np.uint8)
            edge_images.append(binary)
        return np.array(edge_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.ksize}"

# We're testing this to determine if detecting areas of rapid intensity change
# in all directions of the spectrogram provides a comprehensive view of the
# audio signal's structure that improves classification performance.
class LaplacianEdgePreprocessor(ImagePreprocessor):
    def __init__(self, ksize=3):
        self.ksize = ksize

    def process(self, images):
        edge_images = []
        for image in images:
            laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=self.ksize)
            laplacian = np.uint8(np.absolute(laplacian))
            threshold = threshold_otsu(laplacian)
            binary = (laplacian > threshold).astype(np.uint8)
            edge_images.append(binary)
        return np.array(edge_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.ksize}"

# This preprocessor is being evaluated to see if smoothing the spectrogram
# while preserving edges helps reduce noise without losing important
# spectral and temporal transitions, potentially improving the signal-to-noise
# ratio for the Tsetlin Machine.
class BilateralFilterPreprocessor(ImagePreprocessor):
    def __init__(self, d=9, sigmaColor=75, sigmaSpace=75):
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def process(self, images):
        filtered_images = []
        for image in images:
            filtered = cv2.bilateralFilter(image, self.d, self.sigmaColor, self.sigmaSpace)
            threshold = threshold_otsu(filtered)
            binary = (filtered > threshold).astype(np.uint8)
            filtered_images.append(binary)
        return np.array(filtered_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.d}_{self.sigmaColor}_{self.sigmaSpace}"

# We're testing this to assess whether applying thresholding to different
# frequency bands separately allows the Tsetlin Machine to better focus on
# the most relevant spectral components for classification.
class FrequencyBandThresholdingPreprocessor(ImagePreprocessor):
    def __init__(self, num_bands=10):
        self.num_bands = num_bands

    def process(self, images):
        images = self._ensure_3d(images)
        binary_images = []
        for image in images:
            height, width = image.shape
            band_height = height // self.num_bands
            binary_image = np.zeros_like(image, dtype=np.uint8)

            for i in range(self.num_bands):
                start = i * band_height
                end = (i + 1) * band_height if i < self.num_bands - 1 else height
                band = image[start:end, :]
                threshold = threshold_otsu(band)
                binary_image[start:end, :] = (band > threshold).astype(np.uint8)

            binary_images.append(binary_image)
        return np.array(binary_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.num_bands}"

# This is being evaluated to see if highlighting the most prominent frequencies
# at each moment helps the Tsetlin Machine identify characteristic harmonic
# structures or dominant tones that are key to distinguishing between classes.
class PeakDetectionPreprocessor(ImagePreprocessor):
    def __init__(self, min_distance=10):
        self.min_distance = min_distance

    def process(self, images):
        binary_images = []
        for image in images:
            peaks = np.zeros_like(image, dtype=np.uint8)
            for col in range(image.shape[1]):
                col_peaks = self._detect_peaks(image[:, col], self.min_distance)
                peaks[col_peaks, col] = 1
            binary_images.append(peaks)
        return np.array(binary_images)

    def _detect_peaks(self, x, min_distance):
        peaks = []
        for i in range(1, len(x) - 1):
            if x[i-1] < x[i] and x[i] > x[i+1]:
                peaks.append(i)

        # Apply minimum distance
        filtered_peaks = [peaks[0]] if peaks else []
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        return filtered_peaks

    def __str__(self):
        return f"{self.__class__.__name__}_{self.min_distance}"

# We're testing this to determine if making the differences between high-energy
# and low-energy components more pronounced helps the Tsetlin Machine distinguish
# between foreground and background elements in the audio.
class SpectralContrastEnhancementPreprocessor(ImagePreprocessor):
    def __init__(self, percentile=95):
        self.percentile = percentile

    def process(self, images):
        enhanced_images = []
        for image in images:
            threshold = np.percentile(image, self.percentile)
            enhanced = (image > threshold).astype(np.uint8)
            enhanced_images.append(enhanced)
        return np.array(enhanced_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.percentile}"

# This preprocessor is being evaluated to see if highlighting boundaries in
# both time and frequency dimensions helps the Tsetlin Machine identify key
# events and transitions in the audio signal that are crucial for classification.
class TimeFrequencyEdgeDetectionPreprocessor(ImagePreprocessor):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def process(self, images):
        images = self._ensure_3d(images)
        edge_images = []
        for image in images:
            edges = canny(image, sigma=self.sigma)
            edge_images.append(edges.astype(np.uint8))
        return np.array(edge_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.sigma}"

# We're testing this to assess whether emphasizing harmonic structures in the
# spectrogram provides the Tsetlin Machine with strong features for distinguishing
# between different types of harmonic sounds.
class HarmonicStructureEnhancementPreprocessor(ImagePreprocessor):
    def __init__(self, min_angle=80, max_angle=100):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def process(self, images):
        images = self._ensure_3d(images)
        enhanced_images = []
        for image in images:
            edges = canny(image)
            h, theta, d = hough_line(edges)
            lines = []
            for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
                if self.min_angle < np.degrees(angle) < self.max_angle:
                    lines.append((angle, dist))

            line_image = np.zeros_like(image, dtype=np.uint8)
            for angle, dist in lines:
                y0, y1 = (dist - 0 * np.cos(angle)) / np.sin(angle), (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
                line_image[int(y0):int(y1), :] = 1

            enhanced_images.append(line_image)
        return np.array(enhanced_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.min_angle}_{self.max_angle}"

# This is being evaluated to see if combining adaptive thresholding with
# skeletonization helps highlight core spectro-temporal patterns while
# removing less significant details, potentially making it easier for the
# Tsetlin Machine to identify key discriminative features.
class AdaptiveThresholdingWithSkeletonizationPreprocessor(ImagePreprocessor):
    def __init__(self, block_size=35, offset=10):
        self.block_size = block_size
        self.offset = offset

    def process(self, images):
        processed_images = []
        for image in images:
            threshold = threshold_local(image, block_size=self.block_size, offset=self.offset)
            binary = (image > threshold).astype(np.uint8)
            skeleton = skeletonize(binary)
            processed_images.append(skeleton.astype(np.uint8))
        return np.array(processed_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.block_size}_{self.offset}"