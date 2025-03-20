"""
Face and eye detection module using MediaPipe and OpenCV.
"""
from typing import List, Tuple, Optional, Dict
import logging
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

from ..utils.config import detection_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EyeLocation:
    """Data class to store eye location information."""
    left: Tuple[int, int]  # (x, y) center coordinates
    right: Tuple[int, int]
    left_size: Tuple[int, int]  # (width, height)
    right_size: Tuple[int, int]
    face_size: Tuple[int, int]  # (width, height) of the face
    face_center: Tuple[int, int]  # (x, y) center of the face
    confidence: float = 1.0  # Detection confidence score
    rotation: float = 0.0  # Face rotation in degrees

class FaceDetector:
    """Face and eye detection using MediaPipe with OpenCV fallback."""
    
    def __init__(self):
        """Initialize the face detector with MediaPipe and OpenCV cascades."""
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=detection_config.FACE_DETECTION_CONFIDENCE,
            model_selection=1  # Use the full range model for movie posters (good for varied face sizes)
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=detection_config.MAX_FACES,
            min_detection_confidence=detection_config.LANDMARK_DETECTION_CONFIDENCE,
            refine_landmarks=True  # Better eye detection
        )
        
        # Initialize OpenCV Haar cascades as fallback
        if detection_config.USE_HAAR_FALLBACK:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            # Add additional cascade for profile faces
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )

    def _calculate_face_rotation(self, left_eye: Tuple[int, int], right_eye: Tuple[int, int]) -> float:
        """
        Calculate face rotation angle in degrees based on eye positions.
        
        Args:
            left_eye: (x,y) coordinates of left eye
            right_eye: (x,y) coordinates of right eye
            
        Returns:
            Rotation angle in degrees
        """
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        # Calculate angle in radians and convert to degrees
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        return angle

    def detect_faces_mediapipe(self, image: np.ndarray) -> List[EyeLocation]:
        """
        Detect faces and eyes using MediaPipe.

        Args:
            image: numpy.ndarray in BGR format

        Returns:
            List[EyeLocation]: List of detected eye locations
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # First use FaceDetection to get face bounding boxes
        face_detection_results = self.face_detection.process(image_rgb)
        face_boxes = []
        face_scores = []
        
        if face_detection_results.detections:
            for detection in face_detection_results.detections:
                bounding_box = detection.location_data.relative_bounding_box
                face_x = max(0, int(bounding_box.xmin * width))
                face_y = max(0, int(bounding_box.ymin * height))
                face_w = min(int(bounding_box.width * width), width - face_x)
                face_h = min(int(bounding_box.height * height), height - face_y)
                
                if face_w > 0 and face_h > 0:  # Skip invalid boxes
                    face_boxes.append((face_x, face_y, face_w, face_h))
                    face_scores.append(detection.score[0])
                    logger.info(f"Face detected with dimensions: {face_w}x{face_h}, confidence: {detection.score[0]:.2f}")
        
        # Then use FaceMesh for precise eye landmarks
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return []

        eye_locations = []
        
        # Process each face mesh
        for face_index, face_landmarks in enumerate(results.multi_face_landmarks):
            # MediaPipe face mesh indices for precise eye landmarks
            # Using more accurate inner and outer corners for movie poster faces
            left_eye_indices = [33, 133]   # Left eye corners (inner, outer)
            right_eye_indices = [362, 263] # Right eye corners (inner, outer)
            
            # Get eye coordinates
            left_eye = [(int(face_landmarks.landmark[idx].x * width),
                        int(face_landmarks.landmark[idx].y * height))
                       for idx in left_eye_indices]
            right_eye = [(int(face_landmarks.landmark[idx].x * width),
                         int(face_landmarks.landmark[idx].y * height))
                        for idx in right_eye_indices]
            
            # Calculate eye centers and sizes
            left_center = (
                (left_eye[0][0] + left_eye[1][0]) // 2,
                (left_eye[0][1] + left_eye[1][1]) // 2
            )
            right_center = (
                (right_eye[0][0] + right_eye[1][0]) // 2,
                (right_eye[0][1] + right_eye[1][1]) // 2
            )
            
            left_size = (
                max(abs(left_eye[1][0] - left_eye[0][0]), 10),  # Min width of 10px
                max(abs(left_eye[1][1] - left_eye[0][1]), 8)    # Min height of 8px
            )
            right_size = (
                max(abs(right_eye[1][0] - right_eye[0][0]), 10),
                max(abs(right_eye[1][1] - right_eye[0][1]), 8)
            )
            
            # Calculate eye midpoint (center of eyes)
            eye_midpoint = (
                (left_center[0] + right_center[0]) // 2,
                (left_center[1] + right_center[1]) // 2
            )
            
            # Calculate face rotation based on eye positions
            rotation = self._calculate_face_rotation(left_center, right_center)
            
            # Find the best matching face box for this face mesh
            best_match_idx = -1
            best_match_score = float('inf')
            
            for i, (fx, fy, fw, fh) in enumerate(face_boxes):
                # Calculate face box center
                face_center_x = fx + fw // 2
                face_center_y = fy + fh // 2
                
                # Calculate distance between eye midpoint and face center
                distance = np.sqrt(
                    (eye_midpoint[0] - face_center_x) ** 2 + 
                    (eye_midpoint[1] - face_center_y) ** 2
                )
                
                # Normalize by face size for better comparison
                normalized_distance = distance / (fw + fh)
                
                # Adjust for movie poster faces - allow greater distance
                distance_threshold = 0.4 if detection_config.MOVIE_POSTER_MODE else 0.3
                
                # If eye midpoint is within reasonable distance of face center
                if normalized_distance < distance_threshold:
                    if normalized_distance < best_match_score:
                        best_match_score = normalized_distance
                        best_match_idx = i
            
            confidence = 1.0
            
            # If we found a matching face box
            if best_match_idx >= 0:
                face_x, face_y, face_w, face_h = face_boxes[best_match_idx]
                face_size = (face_w, face_h)
                face_center = (face_x + face_w // 2, face_y + face_h // 2)
                confidence = face_scores[best_match_idx]
                
                # Remove the face box so it's not matched again
                face_boxes.pop(best_match_idx)
                face_scores.pop(best_match_idx)
                
                logger.info(f"Matched face mesh {face_index} with face box {best_match_idx}")
            else:
                # For movie posters, be more lenient with floating eyes
                if not detection_config.MOVIE_POSTER_MODE and len(eye_locations) < len(face_boxes):
                    logger.warning(f"Skipping face mesh {face_index} - no matching face box")
                    continue
                
                # Estimate face size based on eye positions
                face_width = abs(right_center[0] - left_center[0]) * 3  # Approx 3x the inter-eye distance
                face_height = face_width * 1.4  # Typical face aspect ratio
                face_size = (int(face_width), int(face_height))
                
                # Ensure minimum face size
                face_size = (max(face_size[0], 100), max(face_size[1], 100))
                
                # Estimate face center
                eye_midpoint_y = (left_center[1] + right_center[1]) // 2
                # Eyes are typically in the upper half of the face
                face_center = (eye_midpoint[0], int(eye_midpoint_y + face_height * 0.15))
                
                # For movie posters, increase the confidence of estimated faces
                confidence = 0.6 if detection_config.MOVIE_POSTER_MODE else 0.5
                logger.info(f"Using estimated face size for face mesh {face_index}")
            
            eye_locations.append(EyeLocation(
                left=left_center,
                right=right_center,
                left_size=left_size,
                right_size=right_size,
                face_size=face_size,
                face_center=face_center,
                confidence=confidence,
                rotation=rotation
            ))
        
        # Sort by confidence (highest first)
        eye_locations.sort(key=lambda x: x.confidence, reverse=True)
        
        # Only return eye locations up to the number of real face detections 
        # plus safety margin (more lenient for movie posters)
        num_detections = 0
        if face_detection_results.detections is not None:
            num_detections = len(face_detection_results.detections)
            
        max_faces = num_detections * 2 if detection_config.MOVIE_POSTER_MODE else num_detections + 1
        max_faces = max(max_faces, 2)  # Always allow at least 2 faces
        
        if len(eye_locations) > max_faces:
            logger.warning(f"Limiting detected faces from {len(eye_locations)} to {max_faces}")
            eye_locations = eye_locations[:max_faces]
            
        return eye_locations

    def detect_faces_opencv(self, image: np.ndarray) -> List[EyeLocation]:
        """
        Detect faces and eyes using OpenCV Haar cascades (fallback method).

        Args:
            image: numpy.ndarray in BGR format

        Returns:
            List[EyeLocation]: List of detected eye locations
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # For movie posters, also try to detect profile faces
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
        
        if detection_config.MOVIE_POSTER_MODE and len(faces) < 2:
            # Try to detect profile faces (both left and right profiles)
            profile_faces = self.profile_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
            # Also try the flipped image to detect faces looking the other way
            flipped = cv2.flip(gray, 1)
            flipped_profile_faces = self.profile_cascade.detectMultiScale(flipped, 1.2, 5, minSize=(30, 30))
            
            # Convert flipped coordinates back to original image
            width = gray.shape[1]
            for i, (x, y, w, h) in enumerate(flipped_profile_faces):
                flipped_profile_faces[i] = (width - x - w, y, w, h)
                
            # Combine all detected faces
            all_faces = list(faces)
            all_faces.extend(profile_faces)
            all_faces.extend(flipped_profile_faces)
            faces = np.array(all_faces) if all_faces else faces
        
        eye_locations = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            
            # Ensure ROI is valid
            if roi_gray.size == 0:
                continue
                
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(10, 10))
            
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate to determine left and right
                eyes = sorted(eyes, key=lambda e: e[0])
                left_eye, right_eye = eyes[:2]
                
                # Convert coordinates to global image space
                left_center = (
                    x + left_eye[0] + left_eye[2]//2,
                    y + left_eye[1] + left_eye[3]//2
                )
                right_center = (
                    x + right_eye[0] + right_eye[2]//2,
                    y + right_eye[1] + right_eye[3]//2
                )
                
                face_size = (w, h)
                face_center = (x + w//2, y + h//2)
                
                # Calculate rotation angle
                rotation = self._calculate_face_rotation(left_center, right_center)
                
                eye_locations.append(EyeLocation(
                    left=left_center,
                    right=right_center,
                    left_size=(max(left_eye[2], 10), max(left_eye[3], 8)),
                    right_size=(max(right_eye[2], 10), max(right_eye[3], 8)),
                    face_size=face_size,
                    face_center=face_center,
                    confidence=0.6 if detection_config.MOVIE_POSTER_MODE else 0.5,  # Higher confidence for movie posters
                    rotation=rotation
                ))
            # Special case for movie posters: if only one eye is detected, estimate the other
            elif detection_config.MOVIE_POSTER_MODE and len(eyes) == 1:
                eye = eyes[0]
                eye_center_x = x + eye[0] + eye[2]//2
                eye_center_y = y + eye[1] + eye[3]//2
                
                # Estimate other eye position based on face width and position
                # If eye is in left half of face, it's the left eye
                if eye[0] < w/2:
                    left_center = (eye_center_x, eye_center_y)
                    # Estimate right eye
                    right_center = (
                        eye_center_x + int(w * 0.4),  # Approx eye separation
                        eye_center_y
                    )
                else:
                    right_center = (eye_center_x, eye_center_y)
                    # Estimate left eye
                    left_center = (
                        eye_center_x - int(w * 0.4),
                        eye_center_y
                    )
                
                face_size = (w, h)
                face_center = (x + w//2, y + h//2)
                
                # Calculate rotation angle
                rotation = self._calculate_face_rotation(left_center, right_center)
                
                eye_locations.append(EyeLocation(
                    left=left_center,
                    right=right_center,
                    left_size=(max(eye[2], 10), max(eye[3], 8)),
                    right_size=(max(eye[2], 10), max(eye[3], 8)),
                    face_size=face_size,
                    face_center=face_center,
                    confidence=0.5,  # Lower confidence for estimated eye
                    rotation=rotation
                ))
        
        return eye_locations

    def detect_eyes(self, image: np.ndarray) -> List[EyeLocation]:
        """
        Detect eyes in an image using MediaPipe with OpenCV fallback.

        Args:
            image: numpy.ndarray in BGR format

        Returns:
            List[EyeLocation]: List of detected eye locations
        """
        # Try MediaPipe first
        eye_locations = self.detect_faces_mediapipe(image)
        
        # Fallback to OpenCV if MediaPipe fails and fallback is enabled
        if not eye_locations and detection_config.USE_HAAR_FALLBACK:
            logger.info("MediaPipe detection failed, falling back to OpenCV")
            eye_locations = self.detect_faces_opencv(image)
            
        # For movie posters, if we still have no eyes, try one more technique
        if not eye_locations and detection_config.MOVIE_POSTER_MODE:
            logger.info("Trying additional movie poster detection technique")
            # Try with equalizing histogram to improve contrast
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            equalized_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            eye_locations = self.detect_faces_mediapipe(equalized_image)
            
            if not eye_locations:
                eye_locations = self.detect_faces_opencv(equalized_image)
        
        return eye_locations 