"""
File handling utilities for loading and saving images.
"""
from pathlib import Path
from typing import Optional, Tuple
import logging
from PIL import Image, ExifTags
import numpy as np
import cv2
import os

from .config import file_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageLoadError(Exception):
    """Exception raised when image loading fails."""
    pass

class ImageSaveError(Exception):
    """Exception raised when image saving fails."""
    pass

def validate_image_path(file_path: str) -> bool:
    """
    Validate if the file path has a supported image format.

    Args:
        file_path: Path to the image file.

    Returns:
        bool: True if the file format is supported, False otherwise.
    """
    return Path(file_path).suffix.lower() in file_config.SUPPORTED_FORMATS

def correct_image_orientation(img: Image.Image) -> Image.Image:
    """
    Correct image orientation based on EXIF data.
    
    Args:
        img: PIL Image to correct
        
    Returns:
        PIL.Image: Corrected image
    """
    try:
        # Find the orientation EXIF tag
        orientation_tag = None
        for tag, value in ExifTags.TAGS.items():
            if value == 'Orientation':
                orientation_tag = tag
                break
                
        if orientation_tag and hasattr(img, '_getexif') and img._getexif():
            exif = dict(img._getexif().items())
            orientation = exif.get(orientation_tag, 1)  # Default is 1 (normal orientation)
            
            # Apply transformations based on orientation value
            if orientation == 1:
                # Normal orientation, no changes needed
                pass
            elif orientation == 2:
                # Mirrored horizontally
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                # Rotated 180 degrees
                img = img.transpose(Image.ROTATE_180)
            elif orientation == 4:
                # Mirrored vertically
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                # Mirrored horizontally and rotated 90 degrees counter-clockwise
                img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
            elif orientation == 6:
                # Rotated 90 degrees counter-clockwise
                img = img.transpose(Image.ROTATE_270)
            elif orientation == 7:
                # Mirrored horizontally and rotated 90 degrees clockwise
                img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
            elif orientation == 8:
                # Rotated 90 degrees clockwise
                img = img.transpose(Image.ROTATE_90)
                
            logger.info(f"Corrected image orientation from EXIF orientation {orientation}")
    except Exception as e:
        logger.warning(f"Could not correct image orientation: {str(e)}")
    
    return img

def load_image(file_path: str) -> Tuple[np.ndarray, str]:
    """
    Load an image file and convert it to a numpy array.

    Args:
        file_path: Path to the image file.

    Returns:
        Tuple containing:
            - numpy.ndarray: Image as a numpy array in RGB or RGBA format
            - str: Original file path

    Raises:
        ImageLoadError: If the image cannot be loaded or is invalid.
    """
    try:
        if not validate_image_path(file_path):
            raise ImageLoadError(f"Unsupported file format. Supported formats: {file_config.SUPPORTED_FORMATS}")

        # Load image with PIL
        with Image.open(file_path) as img:
            # Correct image orientation based on EXIF data
            img = correct_image_orientation(img)
            
            # Handle transparency for PNG images
            if img.format == 'PNG' and 'A' in img.getbands():
                # Convert to RGBA
                img = img.convert('RGBA')
                # Convert to numpy array (already in RGBA format)
                img_array = np.array(img)
                # No channel swapping needed - keep as RGBA
            else:
                # Convert to RGB for non-transparent images
                img = img.convert('RGB')
                # Convert to numpy array (already in RGB format)
                img_array = np.array(img)
            
            # Resize if necessary
            if img.size[0] > file_config.MAX_IMAGE_SIZE[0] or img.size[1] > file_config.MAX_IMAGE_SIZE[1]:
                scale = min(file_config.MAX_IMAGE_SIZE[0] / img.size[0],
                          file_config.MAX_IMAGE_SIZE[1] / img.size[1])
                new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                
                # Use PIL resize to maintain RGB format
                img = img.resize(new_size, Image.LANCZOS)
                img_array = np.array(img)
            
            return img_array, file_path

    except Exception as e:
        raise ImageLoadError(f"Failed to load image: {str(e)}")

def get_output_path(original_path: str) -> str:
    """
    Generate an output file path in the output directory with proper naming.
    
    Args:
        original_path: Path to the original image file
        
    Returns:
        str: Path where the image should be saved
    """
    # Extract the base filename without path and extension
    original_filename = Path(original_path).stem
    original_extension = Path(original_path).suffix
    
    # Create the base output filename with suffix
    output_filename = f"{original_filename}{file_config.OUTPUT_SUFFIX}{original_extension}"
    output_path = os.path.join(file_config.OUTPUT_DIR, output_filename)
    
    # If file already exists, add a number suffix
    counter = 1
    while os.path.exists(output_path):
        output_filename = f"{original_filename}{file_config.OUTPUT_SUFFIX}_{counter}{original_extension}"
        output_path = os.path.join(file_config.OUTPUT_DIR, output_filename)
        counter += 1
    
    return output_path

def save_image(
    img: np.ndarray,
    output_path: Optional[str] = None,
    original_path: Optional[str] = None
) -> str:
    """
    Save a numpy array as an image file.

    Args:
        img: numpy.ndarray image in RGB or RGBA format
        output_path: Optional path where to save the image. If None, a path will be generated.
        original_path: Optional original path of the image, used for generating the output path.

    Returns:
        str: Path where the image was saved.

    Raises:
        ImageSaveError: If the image cannot be saved.
    """
    try:
        if output_path is None:
            if original_path is None:
                raise ImageSaveError("Either output_path or original_path must be provided.")
            output_path = get_output_path(original_path)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Get file extension
        _, extension = os.path.splitext(output_path)
        extension = extension.lower()

        # Handle transparency for PNG files
        if extension == '.png' and img.shape[2] == 4:  # RGBA
            # Convert from RGBA to BGR for OpenCV
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(output_path, bgr)
        else:
            # Convert from RGB to BGR for OpenCV
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, bgr)

        # Check if file was actually created
        if not os.path.exists(output_path):
            raise ImageSaveError(f"Failed to save image to {output_path}.")

        logger.info(f"Image saved to {output_path}")
        return output_path

    except Exception as e:
        raise ImageSaveError(f"Failed to save image: {str(e)}")

def create_preview(img: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    """
    Create a preview of the image scaled to the specified size.

    Args:
        img: numpy.ndarray in BGR or BGRA format
        size: Tuple of (width, height) for the preview

    Returns:
        PIL.Image: Preview image
    """
    # Convert BGR(A) to RGB(A)
    if img.shape[2] == 4:  # BGRA
        img_rgb = img.copy()
        img_rgb[:, :, [0, 2]] = img_rgb[:, :, [2, 0]]  # BGRA to RGBA
        pil_img = Image.fromarray(img_rgb, 'RGBA')
    else:  # BGR
        img_rgb = img[:, :, ::-1]
        pil_img = Image.fromarray(img_rgb, 'RGB')
    
    # Create preview
    pil_img.thumbnail(size)
    
    return pil_img 