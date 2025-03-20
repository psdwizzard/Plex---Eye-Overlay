"""
Image overlay processor for applying eye overlays.
"""
from typing import Tuple, List
import cv2
import numpy as np
from .detector import EyeLocation
from ..utils.config import detection_config
import logging

# Configure logging
logger = logging.getLogger(__name__)

def resize_overlay(
    overlay: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Resize the overlay image while maintaining aspect ratio.

    Args:
        overlay: numpy.ndarray of the overlay image in RGBA format
        target_size: Tuple of (width, height) for the target size

    Returns:
        numpy.ndarray: Resized overlay image in RGBA format
    """
    h, w = overlay.shape[:2]
    target_w, target_h = target_size
    
    # Ensure minimum target size
    target_w = max(target_w, 5)
    target_h = max(target_h, 5)
    
    # Calculate scaling factor while maintaining aspect ratio
    scale = min(target_w / w, target_h / h)
    new_size = (int(w * scale), int(h * scale))
    
    # Ensure new size is at least 5x5
    new_size = (max(new_size[0], 5), max(new_size[1], 5))
    
    # Use INTER_AREA for shrinking, INTER_CUBIC for enlarging
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(overlay, new_size, interpolation=interpolation)

def calculate_eye_size(eye_locations: List[EyeLocation]) -> Tuple[int, int]:
    """
    Calculate a consistent eye size to use for all eyes.
    
    Args:
        eye_locations: List of EyeLocation objects containing eye positions
        
    Returns:
        Tuple of (width, height) for the standardized eye size
    """
    if not eye_locations:
        return (50, 30)  # Default size if no eyes detected
    
    # Calculate average eye size
    total_width = 0
    total_height = 0
    count = 0
    
    for eye_loc in eye_locations:
        total_width += eye_loc.left_size[0] + eye_loc.right_size[0]
        total_height += eye_loc.left_size[1] + eye_loc.right_size[1]
        count += 2  # Count both eyes
    
    # Use average size or default if no valid sizes
    if count > 0:
        avg_width = int(total_width / count)
        avg_height = int(total_height / count)
        # Set minimum size to avoid tiny eyes
        avg_width = max(avg_width, 20)
        avg_height = max(avg_height, 15)
        return (avg_width, avg_height)
    else:
        return (50, 30)  # Default size

def calculate_target_size(
    eye_location: EyeLocation,
    eye_index: str,  # "left" or "right"
    standard_eye_size: Tuple[int, int] = None
) -> Tuple[int, int]:
    """
    Calculate the target size for the eye overlay based on configuration.
    
    Args:
        eye_location: EyeLocation object containing eye and face information
        eye_index: Which eye to calculate for ("left" or "right")
        standard_eye_size: Optional standardized eye size
        
    Returns:
        Tuple of (width, height) for the target overlay size
    """
    if detection_config.SCALE_BY_FACE_SIZE:
        # Scale based on face size
        face_width, face_height = eye_location.face_size
        
        # Target width is a percentage of face width
        target_width = int(face_width * detection_config.FACE_BASED_EYE_SCALE)
        
        # Make height proportional to width (assuming roughly circular eyes)
        target_height = target_width
        
        # Ensure minimum size
        target_width = max(target_width, 20)
        target_height = max(target_height, 20)
        
        logger.info(f"Scaling {eye_index} eye based on face size: {face_width}x{face_height} → {target_width}x{target_height}")
        return (target_width, target_height)
    else:
        # Scale based on eye size
        if detection_config.USE_SAME_SIZE_FOR_BOTH_EYES and standard_eye_size is not None:
            size_to_use = standard_eye_size
        else:
            size_to_use = eye_location.left_size if eye_index == "left" else eye_location.right_size
            
        # Calculate target size based on eye size and scale factor
        target_width = int(size_to_use[0] * detection_config.EYE_OVERLAY_SCALE)
        target_height = int(size_to_use[1] * detection_config.EYE_OVERLAY_SCALE)
        
        # Ensure minimum size
        target_width = max(target_width, 20)
        target_height = max(target_height, 20)
        
        return (target_width, target_height)

def apply_overlay(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    eye_location: EyeLocation,
    eye_index: str,  # "left" or "right"
    standard_eye_size: Tuple[int, int] = None
) -> np.ndarray:
    """
    Apply an overlay image at the specified position.

    Args:
        base_image: numpy.ndarray of the base image in RGB format
        overlay_image: numpy.ndarray of the overlay image in RGBA format
        eye_location: EyeLocation containing eye and face information
        eye_index: Which eye to apply overlay to ("left" or "right")
        standard_eye_size: Optional tuple for standardized eye size for all eyes

    Returns:
        numpy.ndarray: Image with overlay applied in RGB format
    """
    try:
        h_base, w_base = base_image.shape[:2]
        
        # Get the correct eye position
        position = eye_location.left if eye_index == "left" else eye_location.right
        
        # Calculate the target size for the overlay
        target_width, target_height = calculate_target_size(eye_location, eye_index, standard_eye_size)
        
        # Limit the size to reasonable bounds relative to the image
        max_width = w_base // 2
        max_height = h_base // 2
        if target_width > max_width or target_height > max_height:
            scale = min(max_width / target_width, max_height / target_height)
            target_width = int(target_width * scale)
            target_height = int(target_height * scale)
            
        # Ensure minimum size
        target_width = max(target_width, 10)
        target_height = max(target_height, 10)
        
        # Resize overlay
        overlay_resized = resize_overlay(overlay_image, (target_width, target_height))
        h, w = overlay_resized.shape[:2]
        
        # Calculate overlay position (center-aligned)
        x1 = position[0] - w // 2
        y1 = position[1] - h // 2
        x2 = x1 + w
        y2 = y1 + h
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w_base))
        y1 = max(0, min(y1, h_base))
        x2 = max(0, min(x2, w_base))
        y2 = max(0, min(y2, h_base))
        
        # If the overlay would be completely out of bounds, return original image
        if x1 >= x2 or y1 >= y2:
            return base_image
        
        # Calculate crop dimensions for overlay
        overlay_x1 = 0 if x1 >= 0 else -x1
        overlay_y1 = 0 if y1 >= 0 else -y1
        overlay_x2 = w - (x2 - w_base if x2 > w_base else 0)
        overlay_y2 = h - (y2 - h_base if y2 > h_base else 0)
        
        # Make sure we have valid crop dimensions
        if overlay_x1 >= overlay_x2 or overlay_y1 >= overlay_y2:
            return base_image
            
        # Get the region of interest from the base image
        roi = base_image[y1:y2, x1:x2]
        
        # Crop the overlay
        overlay_crop = overlay_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        
        # Ensure overlay has 4 channels (BGRA)
        if overlay_crop.shape[2] < 4:
            logger.warning(f"Overlay has {overlay_crop.shape[2]} channels, expected 4. Adding alpha channel.")
            # If overlay doesn't have an alpha channel, add one (fully opaque)
            overlay_bgr = overlay_crop
            alpha = np.ones((overlay_crop.shape[0], overlay_crop.shape[1], 1), dtype=np.uint8) * 255
            overlay_crop = np.concatenate([overlay_bgr, alpha], axis=2)
            
        # Extract the alpha channel and normalize it
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=-1)  # Add channel dimension
        
        # Extract the BGR channels
        overlay_bgr = overlay_crop[:, :, :3]
        
        # Make sure the shapes match (may be mismatched due to clipping/rounding)
        if roi.shape[:2] != overlay_bgr.shape[:2]:
            min_h = min(roi.shape[0], overlay_bgr.shape[0])
            min_w = min(roi.shape[1], overlay_bgr.shape[1])
            
            if min_h <= 0 or min_w <= 0:
                logger.warning(f"Invalid ROI dimensions: {roi.shape[:2]}, overlay: {overlay_bgr.shape[:2]}")
                return base_image
                
            roi = roi[:min_h, :min_w]
            overlay_bgr = overlay_bgr[:min_h, :min_w]
            alpha = alpha[:min_h, :min_w]
        
        # Blend the images using the alpha channel
        blended = roi * (1 - alpha) + overlay_bgr * alpha
        
        # Create output image
        result = base_image.copy()
        result[y1:y2, x1:x2] = blended
        
        return result
        
    except Exception as e:
        logger.error(f"Error applying overlay to {eye_index} eye: {str(e)}")
        # If something goes wrong, return the original image
        return base_image

def process_image(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    eye_locations: List[EyeLocation]
) -> np.ndarray:
    """
    Process an image by applying eye overlays to all detected eyes.

    Args:
        base_image: numpy.ndarray of the base image in RGB format
        overlay_image: numpy.ndarray of the overlay image in BGRA or RGBA format
        eye_locations: List of EyeLocation objects containing eye positions

    Returns:
        numpy.ndarray: Processed image with overlays applied in RGB format
    """
    result = base_image.copy()
    
    # Ensure overlay has alpha channel
    if overlay_image.shape[2] == 3:  # RGB/BGR without alpha
        logger.info("Adding alpha channel to overlay image")
        alpha = np.ones((overlay_image.shape[0], overlay_image.shape[1], 1), dtype=np.uint8) * 255
        overlay_image = np.concatenate([overlay_image, alpha], axis=2)
    
    # Filter out low confidence detections if PREVENT_FLOATING_EYES is enabled
    if detection_config.PREVENT_FLOATING_EYES:
        filtered_locations = [
            location for location in eye_locations 
            if location.confidence >= detection_config.MIN_FACE_CONFIDENCE
        ]
        if len(filtered_locations) < len(eye_locations):
            logger.info(
                f"Filtered out {len(eye_locations) - len(filtered_locations)} "
                f"low confidence face detection(s) below threshold {detection_config.MIN_FACE_CONFIDENCE}"
            )
        eye_locations = filtered_locations
    
    # If no valid locations after filtering, return original image
    if not eye_locations:
        logger.warning("No valid face detections found after filtering")
        return result
    
    # Calculate standard eye size for consistent overlays if enabled
    standard_eye_size = calculate_eye_size(eye_locations) if detection_config.USE_SAME_SIZE_FOR_BOTH_EYES else None
    
    for eye_loc in eye_locations:
        try:
            # Apply overlay to left eye
            result = apply_overlay(
                result,
                overlay_image,
                eye_loc,
                "left",
                standard_eye_size
            )
            
            # Apply overlay to right eye
            result = apply_overlay(
                result,
                overlay_image,
                eye_loc,
                "right",
                standard_eye_size
            )
        except Exception as e:
            logger.error(f"Error processing face: {str(e)}")
            # Continue with the next face
            continue
    
    return result 