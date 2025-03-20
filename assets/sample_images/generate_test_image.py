"""
Generate sample test images for the Eye Image Overlay Application.
"""
import cv2
import numpy as np
from pathlib import Path

def create_sample_face(width: int = 800, height: int = 600) -> np.ndarray:
    """Create a sample face image with clear features."""
    # Create base image
    image = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw face
    center = (width // 2, height // 2)
    cv2.ellipse(image, center, (120, 160), 0, 0, 360, (200, 200, 200), -1)  # Face
    
    # Draw eyes
    left_eye_center = (center[0] - 50, center[1] - 30)
    right_eye_center = (center[0] + 50, center[1] - 30)
    
    # Eye whites
    cv2.ellipse(image, left_eye_center, (30, 20), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(image, right_eye_center, (30, 20), 0, 0, 360, (255, 255, 255), -1)
    
    # Eye pupils
    cv2.circle(image, left_eye_center, 8, (50, 50, 50), -1)
    cv2.circle(image, right_eye_center, 8, (50, 50, 50), -1)
    
    # Draw eyebrows
    cv2.line(image, 
             (left_eye_center[0] - 35, left_eye_center[1] - 25),
             (left_eye_center[0] + 35, left_eye_center[1] - 20),
             (100, 100, 100), 5)
    cv2.line(image,
             (right_eye_center[0] - 35, right_eye_center[1] - 20),
             (right_eye_center[0] + 35, right_eye_center[1] - 25),
             (100, 100, 100), 5)
    
    # Draw nose
    nose_points = np.array([
        [center[0], center[1] + 20],
        [center[0] - 20, center[1] + 50],
        [center[0] + 20, center[1] + 50]
    ], np.int32)
    cv2.fillPoly(image, [nose_points], (180, 180, 180))
    
    # Draw mouth
    cv2.ellipse(image,
                (center[0], center[1] + 80),
                (60, 20), 0, 0, 180,
                (150, 100, 100), 3)
    
    return image

def create_sample_eye_overlay(size: int = 100) -> np.ndarray:
    """Create a sample eye overlay image."""
    # Create transparent image
    image = np.zeros((size, size, 4), dtype=np.uint8)
    
    # Draw a star-shaped eye
    center = (size // 2, size // 2)
    radius = size // 3
    points = 5
    inner_radius = radius // 2
    
    # Calculate star points
    angles = np.linspace(0, 2 * np.pi, points * 2, endpoint=False)
    pts = []
    
    for i, angle in enumerate(angles):
        r = radius if i % 2 == 0 else inner_radius
        x = center[0] + int(r * np.cos(angle))
        y = center[1] + int(r * np.sin(angle))
        pts.append([x, y])
    
    # Draw star
    pts = np.array(pts, np.int32)
    cv2.fillPoly(image, [pts], (255, 0, 0, 255))  # Red star
    
    # Add some glow effect
    kernel = np.ones((5, 5), np.uint8)
    image[:, :, 3] = cv2.dilate(image[:, :, 3], kernel, iterations=2)
    
    return image

def main():
    """Generate and save sample images."""
    output_dir = Path(__file__).parent
    
    # Create and save face image
    face_image = create_sample_face()
    cv2.imwrite(str(output_dir / "sample_face.jpg"), face_image)
    
    # Create and save eye overlay
    eye_overlay = create_sample_eye_overlay()
    cv2.imwrite(str(output_dir / "sample_eye_overlay.png"), eye_overlay)
    
    print("Sample images generated successfully!")

if __name__ == "__main__":
    main() 