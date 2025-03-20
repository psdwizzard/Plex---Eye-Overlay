"""
Tests for the face and eye detection module.
"""
import pytest
import numpy as np
import cv2
from src.processors.detector import FaceDetector, EyeLocation

def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test image with a simple face-like pattern."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw face circle
    center = (width // 2, height // 2)
    cv2.circle(image, center, 100, (200, 200, 200), -1)
    
    # Draw eyes
    left_eye = (center[0] - 40, center[1] - 20)
    right_eye = (center[0] + 40, center[1] - 20)
    cv2.circle(image, left_eye, 15, (255, 255, 255), -1)
    cv2.circle(image, right_eye, 15, (255, 255, 255), -1)
    
    return image

def test_face_detector_initialization():
    """Test that FaceDetector can be initialized."""
    detector = FaceDetector()
    assert detector is not None
    assert detector.face_detection is not None
    assert detector.face_mesh is not None

def test_detect_eyes_with_test_image():
    """Test eye detection on a test image."""
    detector = FaceDetector()
    test_image = create_test_image()
    
    # Test detection
    eye_locations = detector.detect_eyes(test_image)
    
    # We may not detect eyes in the test image since it's very simple,
    # but the function should run without errors
    assert isinstance(eye_locations, list)

def test_eye_location_dataclass():
    """Test EyeLocation dataclass functionality."""
    eye_loc = EyeLocation(
        left=(100, 100),
        right=(200, 100),
        left_size=(30, 20),
        right_size=(30, 20)
    )
    
    assert eye_loc.left == (100, 100)
    assert eye_loc.right == (200, 100)
    assert eye_loc.left_size == (30, 20)
    assert eye_loc.right_size == (30, 20)

def test_detect_eyes_with_empty_image():
    """Test that detection handles empty images gracefully."""
    detector = FaceDetector()
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    eye_locations = detector.detect_eyes(empty_image)
    assert isinstance(eye_locations, list)
    assert len(eye_locations) == 0

def test_detect_eyes_with_invalid_input():
    """Test that detection handles invalid input gracefully."""
    detector = FaceDetector()
    
    # Test with None
    with pytest.raises(Exception):
        detector.detect_eyes(None)
    
    # Test with invalid shape
    invalid_image = np.zeros((100, 100), dtype=np.uint8)  # Missing channels
    with pytest.raises(Exception):
        detector.detect_eyes(invalid_image)

if __name__ == "__main__":
    pytest.main([__file__]) 