"""
Plex API integration for the Eye Image Overlay Application.
Handles direct communication with Plex Media Server to fetch and update posters.
"""
import os
import logging
import tempfile
import requests
from io import BytesIO
from plexapi.server import PlexServer
from PIL import Image
import numpy as np
import cv2
import shutil
import time
from typing import Dict, List, Tuple, Optional, Any, Callable

from ..processors.detector import FaceDetector
from ..processors.overlay import process_image
from ..utils.file_handler import load_image, get_output_path
from ..utils.config import plex_config, detection_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlexIntegration:
    """Handles integration with Plex Media Server using PlexAPI."""

    def __init__(self, face_detector):
        """Initialize the Plex integration.
        
        Args:
            face_detector: FaceDetector instance to use for face detection
        """
        self.face_detector = face_detector
        self.temp_dir = tempfile.mkdtemp(prefix="eye_overlay_")
        self.progress_callback = None
        logger.info(f"Created temporary directory for Plex files: {self.temp_dir}")
        
        # Create the backup directory if it doesn't exist
        os.makedirs(plex_config.BACKUP_DIR, exist_ok=True)
        logger.info(f"Using backup directory: {plex_config.BACKUP_DIR}")

    def set_progress_callback(self, callback_func):
        """Set a callback function to report progress.
        
        Args:
            callback_func: Function that takes (current_item, total_items, item_name)
        """
        self.progress_callback = callback_func

    def _report_progress(self, current_item, total_items, item_name=""):
        """Report progress using the callback if available.
        
        Args:
            current_item: Current item number
            total_items: Total number of items
            item_name: Name of the current item
        """
        if self.progress_callback:
            self.progress_callback(current_item, total_items, item_name)

    def connect_to_plex(self, server_url, auth_token):
        """Connect to a Plex Media Server.
        
        Args:
            server_url: Base URL of the Plex server (e.g., 'http://localhost:32400')
            auth_token: Plex authentication token
            
        Returns:
            PlexServer instance if connection successful, None otherwise
        """
        try:
            logger.info(f"Connecting to Plex server at {server_url}")
            plex = PlexServer(server_url, auth_token)
            logger.info(f"Successfully connected to Plex server: {plex.friendlyName}")
            return plex
        except Exception as e:
            logger.error(f"Failed to connect to Plex server: {str(e)}")
            return None

    def get_libraries(self, plex_server, library_types=None):
        """Get Plex libraries of specified types.
        
        Args:
            plex_server: PlexServer instance
            library_types: List of library types to include (e.g., ['movie', 'show']) or None for all
            
        Returns:
            List of library sections matching the specified types
        """
        libraries = []
        try:
            for section in plex_server.library.sections():
                if library_types is None or section.type in library_types:
                    libraries.append(section)
                    logger.info(f"Found library: {section.title} (type: {section.type})")
        except Exception as e:
            logger.error(f"Error retrieving Plex libraries: {str(e)}")
            
        return libraries

    def download_poster(self, item):
        """Download a poster image from a Plex media item.
        
        Args:
            item: Plex media item (movie, show, season, etc.)
            
        Returns:
            Tuple of (numpy array of image, temp file path) or (None, None) if failed
        """
        try:
            if not hasattr(item, 'thumbUrl') or not item.thumbUrl:
                logger.warning(f"No poster available for {item.title}")
                return None, None
                
            # Download the poster
            poster_url = item.thumbUrl
            logger.info(f"Downloading poster for {item.title}: {poster_url}")
            
            # Create a temporary file for this poster
            temp_file = os.path.join(self.temp_dir, f"{item.ratingKey}.jpg")
            
            # Download and save the image
            response = requests.get(poster_url, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to download poster: HTTP {response.status_code}")
                return None, None
                
            # Save to temp file
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            try:
                # Try to open with PIL first (handles more formats)
                pil_img = Image.open(BytesIO(response.content))
                # Convert to RGB if needed
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img = np.array(pil_img)
                
                # Ensure the image has valid dimensions
                if img.size == 0 or len(img.shape) < 2:
                    logger.error(f"Invalid image dimensions for {item.title}: {img.shape}")
                    return None, None
                
                # Convert format if needed - ensure we're working with RGB for internal processing
                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                
                return img, temp_file
                
            except Exception as e:
                logger.error(f"Error converting image for {item.title}: {str(e)}")
                
                # Fallback method - try with OpenCV directly
                try:
                    logger.info(f"Trying fallback method for {item.title}")
                    img_array = np.frombuffer(response.content, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if img is None or img.size == 0:
                        logger.error(f"Failed to decode image for {item.title}")
                        return None, None
                        
                    # OpenCV loads as BGR, convert to RGB for consistent internal processing
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img, temp_file
                    
                except Exception as e2:
                    logger.error(f"Fallback method also failed for {item.title}: {str(e2)}")
                    return None, None
            
        except Exception as e:
            logger.error(f"Error downloading poster for {item.title}: {str(e)}")
            return None, None

    def upload_poster(self, item, image_path):
        """Upload a poster to a Plex media item.
        
        Args:
            item: Plex media item
            image_path: Path to the image file to upload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Uploading poster for {item.title}")
            
            # Ensure the image has proper color channels before uploading
            # Read in the image with OpenCV which loads as BGR
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image file: {image_path}")
                return False
                
            # Save as JPG with proper color format (no conversion needed, OpenCV will write as BGR)
            temp_upload_path = os.path.join(self.temp_dir, f"{item.ratingKey}_upload.jpg")
            cv2.imwrite(temp_upload_path, img)
            
            # Upload the properly formatted image
            item.uploadPoster(filepath=temp_upload_path)
            logger.info(f"Successfully uploaded poster for {item.title}")
            return True
        except Exception as e:
            logger.error(f"Error uploading poster for {item.title}: {str(e)}")
            return False

    def process_library_items(self, plex, library_names, overlay_img, debug_mode=False, 
                           movie_poster_mode=False, backup_before_processing=True):
        """Process library items in Plex.
        
        Args:
            plex: PlexServer object
            library_names: List of library names to process
            overlay_img: Eye overlay image as numpy array
            debug_mode: Whether to save debug images
            movie_poster_mode: Whether to enable movie poster mode
            backup_before_processing: Whether to backup posters before processing
            
        Returns:
            Dictionary with statistics about the operation
        """
        # Set movie poster mode
        original_movie_poster_mode = detection_config.MOVIE_POSTER_MODE
        detection_config.MOVIE_POSTER_MODE = movie_poster_mode
        
        # Initialize stats
        stats = {"total": 0, "processed": 0, "failed": 0, "no_face": 0, "backed_up": 0}
        current_item = 0
        
        # First pass: count total items to process
        logger.info("Counting items in libraries...")
        all_movies = []
        all_shows = []
        all_seasons = []
        
        for library_name in library_names:
            try:
                section = plex.library.section(library_name)
                if section.type == 'movie' and plex_config.PROCESS_MOVIE_POSTERS:
                    movies = section.all()
                    all_movies.extend(movies)
                    stats["total"] += len(movies)
                elif section.type == 'show':
                    shows = section.all()
                    if plex_config.PROCESS_SHOW_POSTERS:
                        all_shows.extend(shows)
                        stats["total"] += len(shows)
                    if plex_config.PROCESS_SEASON_POSTERS:
                        for show in shows:
                            seasons = show.seasons()
                            all_seasons.extend(seasons)
                            stats["total"] += len(seasons)
            except Exception as e:
                logger.error(f"Error accessing library {library_name}: {str(e)}")
                
        logger.info(f"Found {stats['total']} items to process")
        
        # Always backup posters before processing to ensure we have the raw data
        if backup_before_processing:
            logger.info("Backing up raw poster data before processing...")
            
            # Count total items for backup
            backup_total = len(all_movies) + len(all_shows) + len(all_seasons)
            backup_current = 0
            
            # Backup movies
            for movie in all_movies:
                backup_current += 1
                self._report_progress(backup_current, backup_total, 
                                     f"Backing up poster ({backup_current}/{backup_total}): {movie.title}")
                
                try:
                    # Download directly from Plex thumbUrl
                    if hasattr(movie, 'thumbUrl') and movie.thumbUrl:
                        # Get raw data
                        response = requests.get(movie.thumbUrl, timeout=10)
                        if response.status_code == 200:
                            # Backup the raw data
                            backup_path = self.backup_poster(movie, image_data=response.content)
                            if backup_path:
                                stats["backed_up"] += 1
                        else:
                            logger.error(f"Failed to download poster for backup: HTTP {response.status_code}")
                    else:
                        logger.warning(f"No thumbUrl for movie: {movie.title}")
                except Exception as e:
                    logger.error(f"Error backing up movie {movie.title}: {str(e)}")
            
            # Backup shows - using same pattern as movies
            for show in all_shows:
                backup_current += 1
                self._report_progress(backup_current, backup_total, 
                                     f"Backing up poster ({backup_current}/{backup_total}): {show.title}")
                
                try:
                    if hasattr(show, 'thumbUrl') and show.thumbUrl:
                        response = requests.get(show.thumbUrl, timeout=10)
                        if response.status_code == 200:
                            backup_path = self.backup_poster(show, image_data=response.content)
                            if backup_path:
                                stats["backed_up"] += 1
                    else:
                        logger.warning(f"No thumbUrl for show: {show.title}")
                except Exception as e:
                    logger.error(f"Error backing up show {show.title}: {str(e)}")
            
            # Backup seasons
            for season in all_seasons:
                backup_current += 1
                show_title = season.show().title if hasattr(season, 'show') else "Unknown Show"
                self._report_progress(backup_current, backup_total, 
                                     f"Backing up poster ({backup_current}/{backup_total}): {show_title} - {season.title}")
                
                try:
                    if hasattr(season, 'thumbUrl') and season.thumbUrl:
                        response = requests.get(season.thumbUrl, timeout=10)
                        if response.status_code == 200:
                            backup_path = self.backup_poster(season, image_data=response.content)
                            if backup_path:
                                stats["backed_up"] += 1
                    else:
                        logger.warning(f"No thumbUrl for season: {season.title}")
                except Exception as e:
                    logger.error(f"Error backing up season {season.title}: {str(e)}")
            
            # Reset counter for processing
            current_item = 0
        
        # Process movies
        current_item = self._process_movie_library(all_movies, overlay_img, debug_mode, current_item, stats)
        
        # Process shows
        current_item = self._process_show_library(all_shows, all_seasons, overlay_img, debug_mode, current_item, stats)
        
        # Restore original mode
        detection_config.MOVIE_POSTER_MODE = original_movie_poster_mode
        
        return stats

    def _process_movie_library(self, movies, overlay_img, debug_mode, current_item=0, stats=None):
        """Process all movies in a movie library.
        
        Args:
            movies: List of Plex movie items
            overlay_img: Image to overlay on eyes
            debug_mode: Whether to enable debug mode
            current_item: Current item number for progress reporting
            stats: Statistics dictionary to update
            
        Returns:
            Updated current_item count
        """
        logger.info(f"Processing {len(movies)} movies")
        
        for movie in movies:
            current_item += 1
            stats["total"] += 1
            progress = current_item / len(movies) if len(movies) > 0 else 0
            
            # Report progress
            self._report_progress(current_item, len(movies), f"Movie: {movie.title}")
            
            logger.info(f"Processing movie [{current_item}/{len(movies)}] ({int(progress*100)}%): {movie.title}")
            
            # Download the poster
            img, temp_file = self.download_poster(movie)
            if img is None:
                stats["failed"] += 1
                logger.warning(f"Failed to download poster for movie: {movie.title}")
                continue
                
            # Detect faces
            eye_locations = self.face_detector.detect_eyes(img)
            
            if not eye_locations:
                logger.warning(f"No faces detected in poster for {movie.title}")
                stats["no_face"] += 1
                continue
                
            # Save debug image if enabled
            if debug_mode:
                debug_img = self._draw_debug_info(img, eye_locations)
                debug_path = os.path.join(self.temp_dir, f"{movie.ratingKey}_debug.jpg")
                # OpenCV expects BGR for writing
                cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved debug image to {debug_path}")
            
            # Process image with googly eyes (process_image expects RGB input)
            result_img = process_image(img, overlay_img, eye_locations)
            
            # Save the processed image - convert to BGR for OpenCV
            processed_path = os.path.join(self.temp_dir, f"{movie.ratingKey}_processed.jpg")
            cv2.imwrite(processed_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            logger.debug(f"Saved processed image to {processed_path}")
            
            # Upload the processed image back to Plex
            if self.upload_poster(movie, processed_path):
                stats["processed"] += 1
                logger.info(f"Successfully processed poster for movie: {movie.title}")
            else:
                stats["failed"] += 1
                logger.warning(f"Failed to upload processed poster for movie: {movie.title}")
                
        return current_item

    def _process_show_library(self, shows, seasons, overlay_img, debug_mode, current_item=0, stats=None):
        """Process all shows in a TV show library.
        
        Args:
            shows: List of Plex show items
            seasons: List of Plex season items
            overlay_img: Image to overlay on eyes
            debug_mode: Whether to enable debug mode
            current_item: Current item number for progress reporting
            stats: Statistics dictionary to update
            
        Returns:
            Updated current_item count
        """
        logger.info(f"Processing {len(shows)} shows and {len(seasons)} seasons")
        
        for show in shows:
            current_item += 1
            stats["total"] += 1
            progress = current_item / (len(shows) + len(seasons)) if (len(shows) + len(seasons)) > 0 else 0
            
            # Report progress
            self._report_progress(current_item, len(shows) + len(seasons), f"Show: {show.title}")
            
            logger.info(f"Processing show [{current_item}/{len(shows) + len(seasons)}] ({int(progress*100)}%): {show.title}")
            
            # Download the poster
            img, temp_file = self.download_poster(show)
            if img is not None:
                # Detect faces
                eye_locations = self.face_detector.detect_eyes(img)
                
                if eye_locations:
                    # Save debug image if enabled
                    if debug_mode:
                        debug_img = self._draw_debug_info(img, eye_locations)
                        debug_path = os.path.join(self.temp_dir, f"show_{show.ratingKey}_debug.jpg")
                        # OpenCV expects BGR for writing
                        cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                    
                    # Process image with googly eyes (process_image expects RGB input)
                    result_img = process_image(img, overlay_img, eye_locations)
                    
                    # Save the processed image - convert to BGR for OpenCV
                    processed_path = os.path.join(self.temp_dir, f"show_{show.ratingKey}_processed.jpg")
                    cv2.imwrite(processed_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                    logger.debug(f"Saved processed image to {processed_path}")
                    
                    # Upload the processed image back to Plex
                    if self.upload_poster(show, processed_path):
                        stats["processed"] += 1
                        logger.info(f"Successfully processed poster for show: {show.title}")
                    else:
                        stats["failed"] += 1
                        logger.warning(f"Failed to upload processed poster for show: {show.title}")
                else:
                    logger.warning(f"No faces detected in poster for show {show.title}")
                    stats["no_face"] += 1
            else:
                stats["failed"] += 1
                logger.warning(f"Failed to download poster for show: {show.title}")
                
            # Process season posters
            for season in seasons:
                current_item += 1
                stats["total"] += 1
                progress = current_item / (len(shows) + len(seasons)) if (len(shows) + len(seasons)) > 0 else 0
                
                # Report progress
                self._report_progress(current_item, len(shows) + len(seasons), f"Season: {season.title} ({show.title})")
                
                logger.info(f"Processing season [{current_item}/{len(shows) + len(seasons)}] ({int(progress*100)}%): {season.title} for show {show.title}")
                
                # Download the poster
                img, temp_file = self.download_poster(season)
                if img is None:
                    stats["failed"] += 1
                    logger.warning(f"Failed to download poster for season: {season.title}")
                    continue
                    
                # Detect faces
                eye_locations = self.face_detector.detect_eyes(img)
                
                if not eye_locations:
                    logger.warning(f"No faces detected in poster for {season.title}")
                    stats["no_face"] += 1
                    continue
                    
                # Save debug image if enabled
                if debug_mode:
                    debug_img = self._draw_debug_info(img, eye_locations)
                    debug_path = os.path.join(self.temp_dir, f"season_{season.ratingKey}_debug.jpg")
                    # OpenCV expects BGR for writing
                    cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                
                # Process image with googly eyes (process_image expects RGB input)
                result_img = process_image(img, overlay_img, eye_locations)
                
                # Save the processed image - convert to BGR for OpenCV
                processed_path = os.path.join(self.temp_dir, f"season_{season.ratingKey}_processed.jpg")
                cv2.imwrite(processed_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                logger.debug(f"Saved processed image to {processed_path}")
                
                # Upload the processed image back to Plex
                if self.upload_poster(season, processed_path):
                    stats["processed"] += 1
                    logger.info(f"Successfully processed poster for season: {season.title}")
                else:
                    stats["failed"] += 1
                    logger.warning(f"Failed to upload processed poster for season: {season.title}")
                    
        return current_item

    def _draw_debug_info(self, image, eye_locations):
        """Draw debug visualization showing face detection results.
        
        Args:
            image: Input image
            eye_locations: List of detected eye locations
            
        Returns:
            Image with debug visualization drawn on it
        """
        debug_image = image.copy()
        
        for i, eye_loc in enumerate(eye_locations):
            # Draw face rectangle
            face_center_x, face_center_y = eye_loc.face_center
            face_width, face_height = eye_loc.face_size
            face_x = face_center_x - face_width // 2
            face_y = face_center_y - face_height // 2
            
            # Determine color based on confidence
            if hasattr(eye_loc, 'confidence') and eye_loc.confidence >= 0.6:  # Default threshold
                face_color = (0, 255, 0)  # Green for high confidence
            else:
                face_color = (0, 0, 255)  # Red for low confidence
                
            # Draw face rectangle
            cv2.rectangle(
                debug_image, 
                (face_x, face_y), 
                (face_x + face_width, face_y + face_height), 
                face_color, 
                2
            )
            
            # Add face confidence text if available
            if hasattr(eye_loc, 'confidence'):
                cv2.putText(
                    debug_image,
                    f"Face {i}: {eye_loc.confidence:.2f}",
                    (face_x, face_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    face_color,
                    1
                )
            
            # Add rotation text
            cv2.putText(
                debug_image,
                f"Rot: {eye_loc.rotation:.1f}°",
                (face_x, face_y + face_height + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                face_color,
                1
            )
            
            # Draw eye centers
            cv2.circle(debug_image, eye_loc.left, 5, (255, 0, 0), -1)  # Left eye (blue)
            cv2.circle(debug_image, eye_loc.right, 5, (255, 0, 0), -1)  # Right eye (blue)
            
            # Draw eye sizes
            left_x, left_y = eye_loc.left
            right_x, right_y = eye_loc.right
            left_w, left_h = eye_loc.left_size
            right_w, right_h = eye_loc.right_size
            
            # Draw rectangles around eyes
            cv2.rectangle(
                debug_image,
                (left_x - left_w//2, left_y - left_h//2),
                (left_x + left_w//2, left_y + left_h//2),
                (255, 0, 0),
                1
            )
            cv2.rectangle(
                debug_image,
                (right_x - right_w//2, right_y - right_h//2),
                (right_x + right_w//2, right_y + right_h//2),
                (255, 0, 0),
                1
            )
            
            # Draw line connecting eyes to show rotation
            cv2.line(debug_image, eye_loc.left, eye_loc.right, (0, 255, 255), 1)
            
        return debug_image

    def cleanup(self):
        """Clean up temporary files."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")

    def backup_poster(self, item, image_data: bytes = None) -> str:
        """Backup a poster from a Plex media item.
        
        Args:
            item: Plex media item (movie, show, season, etc.)
            image_data: Optional image data if already downloaded
            
        Returns:
            Path to the backup file or empty string if failed
        """
        try:
            if not hasattr(item, 'ratingKey'):
                logger.warning(f"Item has no ratingKey, can't backup: {item}")
                return ""
                
            # Create a unique backup filename based on item type and ID
            item_type = getattr(item, 'type', 'unknown')
            backup_filename = f"{item_type}_{item.ratingKey}_{int(time.time())}.raw"
            backup_path = os.path.join(plex_config.BACKUP_DIR, backup_filename)
            
            if image_data:
                # We already have the image data
                with open(backup_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"Backed up poster for {item.title} to {backup_path}")
                return backup_path
                
            # Download the image if we don't have it
            if not hasattr(item, 'thumbUrl') or not item.thumbUrl:
                logger.warning(f"No poster available for {item.title}")
                return ""
                
            # Download the poster directly from Plex
            try:
                logger.info(f"Downloading poster for {item.title} from {item.thumbUrl}")
                response = requests.get(item.thumbUrl, timeout=10)
                if response.status_code != 200:
                    logger.error(f"Failed to download poster for backup: HTTP {response.status_code}")
                    return ""
                
                # Log content type for debugging
                content_type = response.headers.get('Content-Type', '')
                logger.info(f"Content-Type: {content_type} ({len(response.content)} bytes)")
                
                # Save the raw response data exactly as received
                with open(backup_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Backed up raw poster data for {item.title} to {backup_path}")
                return backup_path
                
            except Exception as download_error:
                logger.error(f"Error downloading poster: {str(download_error)}")
                return ""
            
        except Exception as e:
            logger.error(f"Error backing up poster for {item.title}: {str(e)}")
            return ""

    def restore_poster(self, item, backup_path: str) -> bool:
        """Restore a poster from backup to a Plex item.
        
        Args:
            item: Plex item (movie, show, season)
            backup_path: Path to the backup file
            
        Returns:
            bool: True if successful, False otherwise
        """
        item_type = getattr(item, 'type', 'unknown')
        item_id = str(getattr(item, 'ratingKey', 'unknown'))
        item_title = getattr(item, 'title', 'Unknown')
        
        if not os.path.exists(backup_path):
            logger.error(f"Backup file not found: {backup_path}")
            return False
            
        logger.info(f"Restoring poster for {item_type} '{item_title}' (ID: {item_id}) from {backup_path}")
        
        try:
            # Check file size
            file_size = os.path.getsize(backup_path)
            if file_size == 0:
                logger.error(f"Backup file is empty: {backup_path}")
                return False
                
            logger.info(f"Backup file size: {file_size} bytes")
            
            # Open and read the raw binary data
            with open(backup_path, 'rb') as f:
                image_data = f.read()
                
            # No validation or conversion - use the raw data exactly as stored
            logger.info(f"Loaded {len(image_data)} bytes of raw binary data")
            
            # Upload poster to Plex
            try:
                # Check if the item has the uploadPoster method
                if not hasattr(item, 'uploadPoster'):
                    logger.error(f"Item doesn't have uploadPoster method. This may be due to PlexAPI version or permissions.")
                    return False
                    
                # Try different approaches to upload the poster
                logger.info(f"Attempting to upload poster to Plex ({len(image_data)} bytes)")
                
                try:
                    # Attempt 1: Try direct binary upload first (newer PlexAPI versions)
                    item.uploadPoster(image_data)
                    logger.info(f"Successfully restored poster using binary data for {item_title}")
                    return True
                except TypeError as te:
                    # Attempt 2: If binary upload fails, try with filepath (older PlexAPI versions)
                    logger.warning(f"Binary upload failed: {str(te)}, trying with filepath")
                    
                    # Create a temporary file with the original data
                    temp_file = os.path.join(self.temp_dir, f"temp_restore_{item_id}.raw")
                    with open(temp_file, 'wb') as f:
                        f.write(image_data)
                        
                    # Try uploading with filepath
                    item.uploadPoster(filepath=temp_file)
                    logger.info(f"Successfully restored poster using filepath for {item_title}")
                    return True
                except Exception as e:
                    # If both methods fail, try one last approach
                    logger.warning(f"Standard upload methods failed: {str(e)}, trying direct API call")
                    
                    # Attempt 3: Try to directly use the Plex API endpoint
                    # This is a fallback in case the PlexAPI methods don't work
                    try:
                        # Use the underlying PlexAPI connection to make a direct API call
                        headers = {'X-Plex-Token': item._server.token}
                        url = f"{item._server.url(item.key)}/posters"
                        files = {'file': ('poster.raw', image_data)}
                        
                        response = requests.post(url, headers=headers, files=files)
                        if response.status_code < 400:  # Check for success status codes
                            logger.info(f"Successfully restored poster using direct API call for {item_title}")
                            return True
                        else:
                            logger.error(f"Direct API call failed: HTTP {response.status_code}")
                            return False
                    except Exception as api_e:
                        logger.error(f"Direct API call failed: {str(api_e)}")
                        return False
                    
            except Exception as e:
                logger.error(f"Error uploading poster to Plex: {str(e)}")
                # Try to get more details about the error
                error_class = e.__class__.__name__
                if hasattr(e, 'response'):
                    status_code = getattr(e.response, 'status_code', 'unknown')
                    reason = getattr(e.response, 'reason', 'unknown')
                    logger.error(f"HTTP Error: {status_code} - {reason}")
                    
                    # Try to get response content
                    try:
                        content = e.response.text
                        logger.error(f"Response content: {content}")
                    except:
                        pass
                
                # Special handling for common PlexAPI errors
                error_msg = str(e).lower()
                if "missing 1 required positional argument" in error_msg:
                    logger.error("This appears to be a PlexAPI compatibility issue. Your version of PlexAPI may not match the expected version.")
                elif "unauthorized" in error_msg or "permission" in error_msg:
                    logger.error("This appears to be a permissions issue. Make sure your Plex token has write access.")
                elif "not an image" in error_msg:
                    logger.error("Plex rejected the file. It may not be in a format Plex can recognize.")
                        
                return False
                
        except Exception as e:
            logger.error(f"Error restoring poster: {str(e)}")
            return False

    def verify_token_permissions(self, plex):
        """Verify that the Plex token has sufficient permissions.
        
        Args:
            plex: PlexServer object
            
        Returns:
            Tuple of (bool, str) indicating if token has sufficient permissions and a message
        """
        try:
            # Get the server friendly name - if this works, we're connected
            server_name = plex.friendlyName
            logger.info(f"Connected to Plex server: {server_name}")
            
            # Check if we can access libraries
            libraries = plex.library.sections()
            if not libraries:
                return False, "Token has limited permissions - no libraries visible"
            
            logger.info(f"Found {len(libraries)} libraries")
            
            # Look for a test item we can use to check poster access
            for library in libraries:
                if library.type in ('movie', 'show'):
                    test_items = library.search(maxresults=1)
                    if test_items:
                        # We found an item, so we can at least read
                        logger.info(f"Found test item '{test_items[0].title}' in library '{library.title}'")
                        
                        # Try to check if the token has media editing capabilities
                        try:
                            # Check if we have access to the poster URL
                            item = test_items[0]
                            if hasattr(item, 'thumbUrl') and item.thumbUrl:
                                logger.info(f"Item has poster URL: {item.thumbUrl}")
                                return True, f"Connected to server '{server_name}' with library access"
                        except Exception as e:
                            logger.warning(f"Could not access item details: {str(e)}")
            
            # If we got here, we have basic access
            return True, f"Connected to server '{server_name}' with basic access"
                
        except Exception as e:
            logger.error(f"Error verifying token permissions: {str(e)}")
            return False, f"Error verifying token permissions: {str(e)}"
            
    def backup_posters(self, plex, library_names: List[str]) -> Dict[str, int]:
        """Backup all posters from the specified libraries.
        
        Args:
            plex: PlexServer object
            library_names: List of library names to process
            
        Returns:
            Dictionary with statistics about the operation
        """
        stats = {"total": 0, "backed_up": 0, "failed": 0}
        current_item = 0
        
        # First verify token permissions
        has_permission, permission_msg = self.verify_token_permissions(plex)
        logger.info(f"Permission check: {permission_msg}")
        
        if not has_permission:
            logger.error(f"Insufficient permissions to backup posters: {permission_msg}")
            stats["failed"] = 1  # Set to 1 to indicate failure
            stats["error_message"] = permission_msg
            return stats
        
        # First get a count of all items to process
        logger.info("Counting items in libraries...")
        all_movies = []
        all_shows = []
        all_seasons = []
        
        for library_name in library_names:
            try:
                section = plex.library.section(library_name)
                if section.type == 'movie':
                    movies = section.all()
                    all_movies.extend(movies)
                    stats["total"] += len(movies)
                elif section.type == 'show':
                    shows = section.all()
                    all_shows.extend(shows)
                    stats["total"] += len(shows)  # Show posters
                    for show in shows:
                        seasons = show.seasons()
                        all_seasons.extend(seasons)
                        stats["total"] += len(seasons)  # Season posters
            except Exception as e:
                logger.error(f"Error accessing library {library_name}: {str(e)}")
                
        logger.info(f"Found {stats['total']} items to backup")
        
        # Process movies
        for movie in all_movies:
            current_item += 1
            self._report_progress(current_item, stats["total"], f"Backing up movie: {movie.title}")
            
            try:
                backup_path = self.backup_poster(movie)
                if backup_path:
                    stats["backed_up"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Error backing up movie {movie.title}: {str(e)}")
                stats["failed"] += 1
                
        # Process shows
        for show in all_shows:
            current_item += 1
            self._report_progress(current_item, stats["total"], f"Backing up show: {show.title}")
            
            try:
                backup_path = self.backup_poster(show)
                if backup_path:
                    stats["backed_up"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Error backing up show {show.title}: {str(e)}")
                stats["failed"] += 1
                
        # Process seasons
        for season in all_seasons:
            current_item += 1
            show_title = season.show().title if hasattr(season, 'show') else "Unknown Show"
            self._report_progress(current_item, stats["total"], f"Backing up {show_title} - {season.title}")
            
            try:
                backup_path = self.backup_poster(season)
                if backup_path:
                    stats["backed_up"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Error backing up season {season.title}: {str(e)}")
                stats["failed"] += 1
                
        return stats
        
    def restore_posters(self, plex, library_names: List[str]) -> Dict[str, int]:
        """Restore posters for items in the specified libraries from backups.
        
        Args:
            plex: PlexServer object
            library_names: List of library names to process
            
        Returns:
            Dictionary with statistics about the operation
        """
        stats = {"total": 0, "restored": 0, "failed": 0, "no_backup": 0}
        current_item = 0
        
        # First verify connection and basic permissions
        has_connection, connection_msg = self.verify_token_permissions(plex)
        logger.info(f"Connection check: {connection_msg}")
        
        if not has_connection:
            logger.error(f"Cannot connect to Plex server: {connection_msg}")
            stats["failed"] = 1  # Set to 1 to indicate failure
            stats["error_message"] = connection_msg
            return stats
            
        # Next, test if we have restore permissions
        has_restore_permission, restore_msg = self.test_restore_permissions(plex)
        logger.info(f"Restore permission check: {restore_msg}")
        
        if not has_restore_permission:
            logger.error(f"Insufficient permissions to restore posters: {restore_msg}")
            stats["failed"] = 1  # Set to 1 to indicate failure
            stats["error_message"] = restore_msg
            return stats
        
        # List all backup files
        if not os.path.exists(plex_config.BACKUP_DIR):
            logger.error(f"Backup directory not found: {plex_config.BACKUP_DIR}")
            stats["error_message"] = f"Backup directory not found: {plex_config.BACKUP_DIR}"
            return stats
            
        backup_files = [f for f in os.listdir(plex_config.BACKUP_DIR) 
                       if os.path.isfile(os.path.join(plex_config.BACKUP_DIR, f))]
        
        if not backup_files:
            logger.warning(f"No backup files found in {plex_config.BACKUP_DIR}")
            stats["error_message"] = f"No backup files found in {plex_config.BACKUP_DIR}"
            return stats
            
        logger.info(f"Found {len(backup_files)} backup files")
        
        # Parse backup filenames to get item types and IDs
        backup_map = {}
        for filename in backup_files:
            try:
                parts = filename.split('_')
                if len(parts) >= 3:
                    item_type = parts[0]
                    item_id = parts[1]
                    key = f"{item_type}_{item_id}"
                    # Find the newest backup (highest timestamp)
                    if key not in backup_map or filename > backup_map[key]:
                        backup_map[key] = filename
            except Exception as e:
                logger.error(f"Error parsing backup filename {filename}: {str(e)}")
        
        logger.info(f"Found {len(backup_map)} unique items with backups")
        
        # First get a count of all items to process
        logger.info("Counting items in libraries...")
        all_items = []
        
        for library_name in library_names:
            try:
                section = plex.library.section(library_name)
                if section.type == 'movie':
                    all_items.extend(section.all())
                elif section.type == 'show':
                    shows = section.all()
                    all_items.extend(shows)
                    for show in shows:
                        all_items.extend(show.seasons())
            except Exception as e:
                logger.error(f"Error accessing library {library_name}: {str(e)}")
                
        stats["total"] = len(all_items)
        logger.info(f"Found {stats['total']} items to check for restoration")
        
        # Process all items
        for item in all_items:
            current_item += 1
            item_type = getattr(item, 'type', 'unknown')
            item_id = str(item.ratingKey)
            key = f"{item_type}_{item_id}"
            
            self._report_progress(current_item, stats["total"], f"Restoring: {item.title}")
            
            if key in backup_map:
                backup_file = backup_map[key]
                backup_path = os.path.join(plex_config.BACKUP_DIR, backup_file)
                
                try:
                    success = self.restore_poster(item, backup_path)
                    if success:
                        stats["restored"] += 1
                    else:
                        stats["failed"] += 1
                except Exception as e:
                    logger.error(f"Error restoring poster for {item.title}: {str(e)}")
                    stats["failed"] += 1
            else:
                logger.debug(f"No backup found for {item_type} {item.title} (ID: {item_id})")
                stats["no_backup"] += 1
                
        return stats

    def test_restore_permissions(self, plex):
        """Test if the Plex token has sufficient permissions to restore posters.
        
        Args:
            plex: PlexServer object
            
        Returns:
            Tuple of (bool, str) indicating if token has restore permissions and a message
        """
        try:
            # First check if the backup directory exists
            if not os.path.exists(plex_config.BACKUP_DIR):
                return False, f"Backup directory not found: {plex_config.BACKUP_DIR}"
                
            # List all backup files
            backup_files = [f for f in os.listdir(plex_config.BACKUP_DIR) 
                           if os.path.isfile(os.path.join(plex_config.BACKUP_DIR, f))]
            
            if not backup_files:
                return False, f"No backup files found in {plex_config.BACKUP_DIR}"
                
            # Get all libraries
            libraries = plex.library.sections()
            if not libraries:
                return False, "No libraries found"
                
            # Find a test item that has a backup
            for filename in backup_files:
                try:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        item_type = parts[0]
                        item_id = parts[1]
                        
                        # Find this item in the Plex library
                        for library in libraries:
                            if (item_type == 'movie' and library.type == 'movie') or \
                               (item_type in ('show', 'season') and library.type == 'show'):
                                
                                try:
                                    # Try to get the item by ID
                                    item = None
                                    if item_type == 'movie':
                                        # Try to find the movie
                                        for movie in library.all():
                                            if str(movie.ratingKey) == item_id:
                                                item = movie
                                                break
                                    elif item_type == 'show':
                                        # Try to find the show
                                        for show in library.all():
                                            if str(show.ratingKey) == item_id:
                                                item = show
                                                break
                                    elif item_type == 'season':
                                        # Try to find the season
                                        for show in library.all():
                                            for season in show.seasons():
                                                if str(season.ratingKey) == item_id:
                                                    item = season
                                                    break
                                            if item:
                                                break
                                    
                                    if item:
                                        # We found the item, now let's check if we have permission to upload a poster
                                        logger.info(f"Found test item: {item.title} ({item_type}, ID: {item_id})")
                                        
                                        # Check if the item has uploadPoster method
                                        if hasattr(item, 'uploadPoster'):
                                            logger.info("Item has uploadPoster method")
                                            # We're not actually going to call it, just check if it exists
                                            return True, f"Found item '{item.title}' with backup and upload capability"
                                        else:
                                            logger.warning("Item doesn't have uploadPoster method")
                                except Exception as e:
                                    logger.warning(f"Error finding item: {str(e)}")
                                    continue
                except Exception as e:
                    logger.warning(f"Error parsing backup filename {filename}: {str(e)}")
                    continue
                    
            return False, "Could not find any items with backups to test permissions"
                
        except Exception as e:
            return False, f"Error testing restore permissions: {str(e)}" 