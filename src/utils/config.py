"""
Configuration settings for the Eye Image Overlay Application.
"""
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import os

@dataclass
class UIConfig:
    """User interface configuration settings."""
    WINDOW_TITLE: str = "Eye Image Overlay"
    WINDOW_SIZE: Tuple[int, int] = (950, 700)  # Increased size for better layout
    PREVIEW_SIZE: Tuple[int, int] = (400, 400)  # Larger preview size
    PADDING: int = 10
    DARK_MODE: bool = True
    DEFAULT_EYE_OVERLAY: str = str(Path(__file__).parent.parent.parent / "assets" / "eye.png")

@dataclass
class DetectionConfig:
    """Face and eye detection configuration settings."""
    # MediaPipe face detection confidence threshold
    FACE_DETECTION_CONFIDENCE: float = 0.5
    # MediaPipe landmark detection confidence threshold
    LANDMARK_DETECTION_CONFIDENCE: float = 0.5
    
    # Scale googly eyes based on face size instead of eye size
    SCALE_BY_FACE_SIZE: bool = True
    
    # Face-based scale factor (proportion of face width)
    # A value of 0.35 means each eye will be 35% of the face width
    FACE_BASED_EYE_SCALE: float = 0.35
    
    # Eye-based scale factor (used when SCALE_BY_FACE_SIZE is False)
    EYE_OVERLAY_SCALE: float = 40.0
    
    # Both eyes should have the same size overlay
    USE_SAME_SIZE_FOR_BOTH_EYES: bool = True
    
    # Prevent floating eyes by requiring a matching face detection
    PREVENT_FLOATING_EYES: bool = True
    
    # Minimum confidence threshold for face detections
    MIN_FACE_CONFIDENCE: float = 0.6
    
    # Fallback to OpenCV Haar cascades if True and MediaPipe fails
    USE_HAAR_FALLBACK: bool = True
    
    # Movie poster mode for improved detection with odd angles
    MOVIE_POSTER_MODE: bool = False
    
    # Maximum faces to detect in an image
    MAX_FACES: int = 10

@dataclass
class FileConfig:
    """File handling configuration settings."""
    SUPPORTED_FORMATS: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')
    MAX_IMAGE_SIZE: Tuple[int, int] = (2000, 2000)  # Maximum image dimensions
    JPEG_QUALITY: int = 95  # JPEG save quality
    # Output directory for saving processed images
    OUTPUT_DIR: str = str(Path(__file__).parent.parent.parent / "output")
    # File suffix for processed images
    OUTPUT_SUFFIX: str = "_eye"

@dataclass
class PlexConfig:
    """Configuration settings for Plex integration."""
    # Plex server settings
    PLEX_SERVER_URL: str = "http://localhost:32400"  # Change to your Plex server URL
    PLEX_TOKEN: str = ""  # Your Plex authentication token
    
    # Library settings
    MOVIE_LIBRARIES: Tuple[str, ...] = ("Movies",)  # List of movie library names
    TV_LIBRARIES: Tuple[str, ...] = ("TV Shows",)  # List of TV show library names
    
    # Processing options
    PROCESS_SHOW_POSTERS: bool = True  # Process main show posters
    PROCESS_SEASON_POSTERS: bool = True  # Process season posters
    PROCESS_MOVIE_POSTERS: bool = True  # Process movie posters
    
    # Backup options
    BACKUP_DIR: str = str(Path(__file__).parent.parent.parent / "backup_posters")  # Directory for backup files

# Create instances of configuration classes
ui_config = UIConfig()
detection_config = DetectionConfig()
file_config = FileConfig()
plex_config = PlexConfig()

# Ensure output directory exists
os.makedirs(file_config.OUTPUT_DIR, exist_ok=True) 