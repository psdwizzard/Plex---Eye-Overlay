"""
Main application window for the Eye Image Overlay Application.
"""
from typing import Optional, List
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image
import logging
import threading
from pathlib import Path
import os
import cv2
import numpy as np
import glob

from .components import ImagePreview, StatusBar
from ..processors.detector import FaceDetector, EyeLocation
from ..processors.overlay import process_image
from ..utils.file_handler import load_image, save_image, create_preview, ImageLoadError, ImageSaveError, get_output_path
from ..utils.config import ui_config, file_config, detection_config, plex_config
from ..utils.plex_integration import PlexIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppWindow(ctk.CTk):
    """Main application window."""

    def __init__(self):
        """Initialize the application window."""
        super().__init__()

        # Set window properties
        self.title(ui_config.WINDOW_TITLE)
        self.geometry(f"{ui_config.WINDOW_SIZE[0]}x{ui_config.WINDOW_SIZE[1]}")
        
        # Set appearance mode
        ctk.set_appearance_mode("dark" if ui_config.DARK_MODE else "light")
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Create UI components
        self._create_widgets()
        
        # Initialize processing state
        self.processing = False
        self.debug_mode = False
        self.movie_poster_mode = detection_config.MOVIE_POSTER_MODE
        
        # Load default eye overlay
        self._load_default_eye_overlay()
        
        # Batch processing variables
        self.batch_files = []
        self.current_batch_index = 0
        self.batch_total = 0
        self.batch_processed = 0
        self.batch_failed = 0

    def _load_base_image(self, file_path):
        """Load a file as the base image."""
        if not file_path:
            return
            
        try:
            # Check if the file exists and is an image
            if not os.path.isfile(file_path):
                self.status_bar.set_status(f"Not a valid file: {file_path}")
                return
                
            # Check file extension
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                self.status_bar.set_status(f"Not a supported image format: {ext}")
                return
                
            image = Image.open(file_path)
            self.base_preview.update_preview(image, file_path)
            output_path = get_output_path(file_path)
            self.status_bar.set_status(f"Base image loaded: {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Failed to load base image: {str(e)}")
            self.status_bar.set_status(f"Error loading image: {str(e)}")

    def _load_overlay_image(self, file_path):
        """Load a file as the overlay image."""
        if not file_path:
            return
            
        try:
            # Check if the file exists and is an image
            if not os.path.isfile(file_path):
                self.status_bar.set_status(f"Not a valid file: {file_path}")
                return
                
            # Check file extension
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in ['.png', '.jpg', '.jpeg']:
                self.status_bar.set_status(f"Not a supported image format: {ext}")
                return
                
            image = Image.open(file_path)
            self.overlay_preview.update_preview(image, file_path)
            self.status_bar.set_status(f"Overlay image loaded: {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Failed to load overlay image: {str(e)}")
            self.status_bar.set_status(f"Error loading image: {str(e)}")

    def _create_widgets(self):
        """Create and arrange the UI widgets."""
        # Create preview frames
        self.base_preview = ImagePreview(
            self,
            "Base Image",
            "Select Base Image",
            self._select_base_image
        )
        self.base_preview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.overlay_preview = ImagePreview(
            self,
            "Eye Overlay Image",
            "Select Eye Image",
            self._select_overlay_image
        )
        self.overlay_preview.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Create bottom frame for controls and buttons
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        bottom_frame.grid_columnconfigure(0, weight=1)
        
        # Display output directory info
        output_label = ctk.CTkLabel(
            bottom_frame,
            text=f"Output directory: {file_config.OUTPUT_DIR}",
            anchor="w"
        )
        output_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        
        # Create a frame for checkboxes with better spacing
        checkbox_frame = ctk.CTkFrame(bottom_frame)
        checkbox_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        checkbox_frame.grid_columnconfigure(0, weight=1)
        
        # Add debug mode checkbox
        self.debug_var = ctk.BooleanVar(value=False)
        self.debug_checkbox = ctk.CTkCheckBox(
            checkbox_frame,
            text="Debug Mode (show face detection)",
            variable=self.debug_var,
            command=self._toggle_debug_mode
        )
        self.debug_checkbox.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Add movie poster mode checkbox
        self.movie_poster_var = ctk.BooleanVar(value=detection_config.MOVIE_POSTER_MODE)
        self.movie_poster_checkbox = ctk.CTkCheckBox(
            checkbox_frame,
            text="Movie Poster Mode (better detection for posters, angled faces)",
            variable=self.movie_poster_var,
            command=self._toggle_movie_poster_mode
        )
        self.movie_poster_checkbox.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        # Add confidence threshold slider in its own frame
        slider_frame = ctk.CTkFrame(bottom_frame)
        slider_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        slider_frame.grid_columnconfigure(1, weight=1)
        
        conf_label = ctk.CTkLabel(
            slider_frame,
            text="Face Confidence Threshold:",
            anchor="w"
        )
        conf_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.conf_value_label = ctk.CTkLabel(
            slider_frame,
            text=f"{int(detection_config.MIN_FACE_CONFIDENCE * 100)}%",
            width=40
        )
        self.conf_value_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")
        
        self.conf_slider = ctk.CTkSlider(
            slider_frame,
            from_=0,
            to=100,
            number_of_steps=20,
            command=self._update_confidence_threshold
        )
        self.conf_slider.set(detection_config.MIN_FACE_CONFIDENCE * 100)
        self.conf_slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # Create buttons frame
        buttons_frame = ctk.CTkFrame(bottom_frame)
        buttons_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        buttons_frame.grid_columnconfigure(2, weight=1)
        buttons_frame.grid_columnconfigure(3, weight=1)
        
        # Create process button
        self.process_button = ctk.CTkButton(
            buttons_frame,
            text="Add HUGE Googly Eyes",
            command=self._process_image,
            font=("Helvetica", 16, "bold"),
            height=50,
            fg_color="#e5a00d",  # Yellow color
            text_color="black"
        )
        self.process_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        # Create batch process button
        self.batch_button = ctk.CTkButton(
            buttons_frame,
            text="Batch Process Folder",
            command=self._select_folder_for_batch,
            font=("Helvetica", 16, "bold"),
            height=50,
            fg_color="#e5a00d",  # Yellow color
            text_color="black"
        )
        self.batch_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # Create Plex library process button
        self.plex_button = ctk.CTkButton(
            buttons_frame,
            text="Process Plex Library",
            command=self._select_folder_for_plex,
            font=("Helvetica", 16, "bold"),
            height=50,
            fg_color="#e5a00d",  # Yellow color
            text_color="black"
        )
        self.plex_button.grid(row=0, column=2, padx=10, pady=10, sticky="ew")
        
        # Create Process Plex Server button
        self.plex_server_button = ctk.CTkButton(
            buttons_frame,
            text="Process Plex Server",
            command=self._process_plex_server,
            font=("Helvetica", 16, "bold"),
            height=50,
            fg_color="#e5a00d",  # Yellow color
            text_color="black"
        )
        self.plex_server_button.grid(row=0, column=3, padx=10, pady=10, sticky="ew")
        
        # Create status bar
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

    def _update_confidence_threshold(self, value):
        """Update confidence threshold from slider."""
        # Convert slider value (0-100) to confidence (0.0-1.0)
        confidence = round(value / 100, 2)
        detection_config.MIN_FACE_CONFIDENCE = confidence
        self.conf_value_label.configure(text=f"{int(value)}%")
        logger.info(f"Face confidence threshold set to {confidence:.2f}")
        self.status_bar.set_status(f"Confidence threshold set to {int(value)}% - lower values detect more faces but may cause 'floating eyes'")
        
    def _toggle_debug_mode(self):
        """Toggle debug mode on/off."""
        self.debug_mode = self.debug_var.get()
        logger.info(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
        
    def _toggle_movie_poster_mode(self):
        """Toggle movie poster mode on/off."""
        self.movie_poster_mode = self.movie_poster_var.get()
        # Update the global config
        detection_config.MOVIE_POSTER_MODE = self.movie_poster_mode
        logger.info(f"Movie poster mode {'enabled' if self.movie_poster_mode else 'disabled'}")
        
        # Adjust confidence thresholds based on movie poster mode
        if self.movie_poster_mode:
            detection_config.FACE_DETECTION_CONFIDENCE = 0.4
            # Don't change MIN_FACE_CONFIDENCE as it's controlled by the slider
            self.status_bar.set_status("Movie poster mode enabled - better for angled faces and posters")
        else:
            detection_config.FACE_DETECTION_CONFIDENCE = 0.5
            # Don't change MIN_FACE_CONFIDENCE as it's controlled by the slider
            self.status_bar.set_status("Movie poster mode disabled - standard face detection")
        
    def _load_default_eye_overlay(self):
        """Load the default eye overlay image."""
        try:
            if os.path.exists(ui_config.DEFAULT_EYE_OVERLAY):
                image = Image.open(ui_config.DEFAULT_EYE_OVERLAY)
                self.overlay_preview.update_preview(image, ui_config.DEFAULT_EYE_OVERLAY)
                self.status_bar.set_status("Default eye overlay loaded")
                logger.info(f"Default eye overlay loaded from: {ui_config.DEFAULT_EYE_OVERLAY}")
            else:
                error_msg = f"Default eye overlay not found: {ui_config.DEFAULT_EYE_OVERLAY}"
                logger.warning(error_msg)
                self.status_bar.set_status("Default eye overlay not found")
                messagebox.showwarning(
                    "Missing Default Image", 
                    f"Could not find the default eye image at:\n{ui_config.DEFAULT_EYE_OVERLAY}\n\nPlease select an eye image manually."
                )
        except Exception as e:
            logger.error(f"Failed to load default eye overlay: {str(e)}")
            self.status_bar.set_status("Failed to load default eye overlay")

    def _select_base_image(self):
        """Handle base image selection."""
        file_path = filedialog.askopenfilename(
            title="Select Base Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            self._load_base_image(file_path)

    def _select_overlay_image(self):
        """Handle overlay image selection."""
        file_path = filedialog.askopenfilename(
            title="Select Eye Overlay Image",
            filetypes=[("PNG files", "*.png"), ("All image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            self._load_overlay_image(file_path)

    def _select_folder_for_batch(self):
        """Select folder for batch processing."""
        if self.processing:
            self.status_bar.set_status("Processing already in progress")
            return
            
        if not self.overlay_preview.current_image_path:
            self.status_bar.set_status("Please select an overlay image first")
            return
        
        folder_path = filedialog.askdirectory(
            title="Select Folder with Images to Process"
        )
        
        if not folder_path:
            return
            
        # Find all supported image files in the folder
        image_files = []
        for ext in file_config.SUPPORTED_FORMATS:
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            
        if not image_files:
            self.status_bar.set_status(f"No supported image files found in {folder_path}")
            messagebox.showinfo(
                "No Images Found", 
                f"No supported image files found in the selected folder.\n\nSupported formats: {', '.join(file_config.SUPPORTED_FORMATS)}"
            )
            return
            
        # Confirm batch processing
        result = messagebox.askokcancel(
            "Batch Processing", 
            f"Found {len(image_files)} images to process.\n\nThis will apply googly eyes to all images in the folder.\nContinue?"
        )
        
        if not result:
            return
            
        # Start batch processing
        self.batch_files = image_files
        self.batch_total = len(image_files)
        self.batch_processed = 0
        self.batch_failed = 0
        self.current_batch_index = 0
        
        # Start processing in a thread
        self.processing = True
        self.process_button.configure(state="disabled")
        self.batch_button.configure(state="disabled")
        self.status_bar.show_progress()
        
        thread = threading.Thread(target=self._batch_process_thread)
        thread.start()

    def _select_folder_for_plex(self):
        """Select folder for Plex library processing (recursive with overwrite)."""
        if self.processing:
            self.status_bar.set_status("Processing already in progress")
            return
            
        if not self.overlay_preview.current_image_path:
            self.status_bar.set_status("Please select an overlay image first")
            return
        
        folder_path = filedialog.askdirectory(
            title="Select Plex Library Folder"
        )
        
        if not folder_path:
            return
        
        # Confirm Plex processing with overwrite warning
        result = messagebox.askokcancel(
            "Process Plex Library", 
            f"This will recursively process all images in '{folder_path}' and all subfolders.\n\n" + 
            "WARNING: Original images will be OVERWRITTEN with googly eye versions.\n\n" +
            "Are you sure you want to continue?",
            icon="warning"
        )
        
        if not result:
            return
        
        # Start processing in a thread
        self.processing = True
        self.process_button.configure(state="disabled")
        self.batch_button.configure(state="disabled")
        self.plex_button.configure(state="disabled")
        self.status_bar.show_progress()
        
        thread = threading.Thread(target=lambda: self._plex_process_thread(folder_path))
        thread.start()
    
    def _find_all_images(self, root_folder):
        """Recursively find all supported images in a folder and its subfolders, including Plex files without extensions."""
        image_files = []
        for root, _, files in os.walk(root_folder):
            for file in files:
                file_path = os.path.join(root, file)
                
                # First check if it has a supported extension
                _, ext = os.path.splitext(file)
                if ext.lower() in file_config.SUPPORTED_FORMATS:
                    image_files.append(file_path)
                    continue
                
                # Check if it might be a Plex metadata file (without extension)
                if not ext:
                    # Check for known Plex file patterns
                    is_plex_file = False
                    
                    # Check common Plex poster paths
                    if "Uploads/posters" in file_path:
                        is_plex_file = True
                    
                    # Check for Plex agent patterns
                    plex_prefixes = [
                        "tv.plex.agents.", 
                        "com.plexapp.agents.",
                        "plex.agents.",
                        "metadata.plex."
                    ]
                    
                    if any(file.startswith(prefix) for prefix in plex_prefixes):
                        is_plex_file = True
                        
                    # If it looks like a Plex file, try to read it as an image
                    if is_plex_file:
                        try:
                            # Try to read the file as an image
                            with open(file_path, 'rb') as f:
                                header = f.read(12)  # Read first 12 bytes to check signatures
                                
                                # Check for JPEG signature (FF D8 FF)
                                if header.startswith(b'\xFF\xD8\xFF'):
                                    logger.info(f"Found Plex JPEG file: {file_path}")
                                    image_files.append(file_path)
                                    continue
                                    
                                # Check for PNG signature (89 50 4E 47 0D 0A 1A 0A)
                                if header.startswith(b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'):
                                    logger.info(f"Found Plex PNG file: {file_path}")
                                    image_files.append(file_path)
                                    continue
                        except Exception as e:
                            logger.warning(f"Error checking file type for {file_path}: {str(e)}")
                        
        return image_files
        
    def _plex_process_thread(self, root_folder):
        """Thread for processing Plex library images (recursive with overwrite)."""
        try:
            # Find all images in the folder and subfolders
            self.status_bar.set_status("Finding all images in Plex library...")
            image_files = self._find_all_images(root_folder)
            
            if not image_files:
                self.status_bar.set_status("No supported image files found in Plex library")
                messagebox.showinfo(
                    "No Images Found", 
                    f"No supported image files found in the selected folder or its subfolders.\n\n"
                    f"Supported formats: {', '.join(file_config.SUPPORTED_FORMATS)}\n\n"
                    f"Also checked for Plex metadata files (extension-less files in 'Uploads/posters' "
                    f"and files starting with 'tv.plex.agents.', 'com.plexapp.agents.', etc.)"
                )
                return
                
            # Get confirmation with count
            result = messagebox.askokcancel(
                "Plex Library Processing", 
                f"Found {len(image_files)} images to process.\n\n" +
                "This will apply googly eyes to all images and OVERWRITE the original files.\n" +
                "This operation cannot be undone.\n\n" +
                "Continue?",
                icon="warning"
            )
            
            if not result:
                return
            
            # Load overlay image once
            overlay_img, _ = load_image(self.overlay_preview.current_image_path)
            
            # Setup counters
            total = len(image_files)
            processed = 0
            failed = 0
            
            for i, file_path in enumerate(image_files):
                progress = (i + 0.5) / total
                
                try:
                    # Update status
                    filename = os.path.basename(file_path)
                    # Check if this is a Plex file (no extension and either in posters path or has agent prefix)
                    is_plex_file = (not os.path.splitext(file_path)[1]) and (
                        "Uploads/posters" in file_path or
                        any(filename.startswith(prefix) for prefix in ["tv.plex.agents.", "com.plexapp.agents.", "plex.agents."])
                    )
                    
                    self.status_bar.set_status(f"Processing {i+1}/{total}: {filename} {'(Plex Metadata)' if is_plex_file else ''}")
                    self.status_bar.update_progress(progress)
                    
                    # Load base image
                    # For Plex files, detect format from file content
                    if is_plex_file:
                        # We need to detect type of image from content
                        with open(file_path, 'rb') as f:
                            header = f.read(12)
                            if header.startswith(b'\xFF\xD8\xFF'):
                                mime_type = 'jpeg'
                            elif header.startswith(b'\x89\x50\x4E\x47'):
                                mime_type = 'png'
                            else:
                                logger.warning(f"Unrecognized image format for {file_path}")
                                failed += 1
                                continue
                        
                        # Read with PIL directly
                        base_img = np.array(Image.open(file_path))
                        if len(base_img.shape) == 2:  # Grayscale
                            base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
                        elif base_img.shape[2] == 4:  # RGBA
                            base_img = cv2.cvtColor(base_img, cv2.COLOR_RGBA2RGB)
                    else:
                        # Use normal loader for regular files
                        base_img, _ = load_image(file_path)
                    
                    # Update preview if possible
                    try:
                        preview = Image.open(file_path)
                        # Use after to schedule UI update from thread
                        self.after(0, lambda img=preview, path=file_path: 
                                   self.base_preview.update_preview(img, path))
                    except Exception as e:
                        logger.warning(f"Could not update preview for {file_path}: {str(e)}")
                    
                    # Detect eyes
                    eye_locations = self.face_detector.detect_eyes(base_img)
                    
                    if not eye_locations:
                        logger.warning(f"No faces detected in {filename}")
                        failed += 1
                        continue
                    
                    # Save debug image if enabled
                    if self.debug_mode:
                        debug_img = self._draw_debug_info(base_img, eye_locations)
                        if is_plex_file:
                            # Save to output folder with special name
                            debug_path = os.path.join(file_config.OUTPUT_DIR, f"plex_metadata_debug_{filename}.jpg")
                        else:
                            debug_path = get_output_path(file_path).replace("_eye", "_debug")
                        save_image(debug_img, debug_path, None)
                    
                    # Process image
                    result_img = process_image(base_img, overlay_img, eye_locations)
                    
                    # Overwrite the original image
                    if is_plex_file:
                        # For Plex files, we need to save in the original format
                        if mime_type == 'jpeg':
                            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                        else:  # PNG
                            _, buffer = cv2.imencode('.png', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                        with open(file_path, 'wb') as f:
                            f.write(buffer)
                    else:
                        # For regular files use standard cv2.imwrite
                        cv2.imwrite(file_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                    
                    processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    failed += 1
            
            # Final status update
            self.status_bar.update_progress(1.0)
            self.status_bar.set_status(
                f"Plex library processing complete: {processed} processed, "
                f"{failed} failed, {total} total"
            )
            
            # Show completion message
            self.after(0, lambda: messagebox.showinfo(
                "Plex Library Processing Complete",
                f"Processed {processed} images\n"
                f"Failed: {failed} images\n\n"
                f"All successful images have been overwritten with googly eye versions."
            ))
            
        except Exception as e:
            logger.error(f"Plex library processing error: {str(e)}")
            self.status_bar.set_status(f"Plex library processing error: {str(e)}")
        finally:
            self.processing = False
            # Use after to update UI from thread
            self.after(0, lambda: self.process_button.configure(state="normal"))
            self.after(0, lambda: self.batch_button.configure(state="normal"))
            self.after(0, lambda: self.plex_button.configure(state="normal"))

    def _batch_process_thread(self):
        """Thread for batch processing images."""
        try:
            # Load overlay image once
            overlay_img, _ = load_image(self.overlay_preview.current_image_path)
            
            for i, file_path in enumerate(self.batch_files):
                self.current_batch_index = i
                progress = (i + 0.5) / self.batch_total
                
                try:
                    # Update status
                    filename = os.path.basename(file_path)
                    self.status_bar.set_status(f"Processing {i+1}/{self.batch_total}: {filename}")
                    self.status_bar.update_progress(progress)
                    
                    # Load base image
                    base_img, base_path = load_image(file_path)
                    
                    # Update preview if possible
                    try:
                        preview = Image.open(file_path)
                        # Use after to schedule UI update from thread
                        self.after(0, lambda img=preview, path=file_path: 
                                   self.base_preview.update_preview(img, path))
                    except:
                        pass
                    
                    # Detect eyes
                    eye_locations = self.face_detector.detect_eyes(base_img)
                    
                    if not eye_locations:
                        logger.warning(f"No faces detected in {filename}")
                        self.batch_failed += 1
                        continue
                    
                    # Save debug image if enabled
                    if self.debug_mode:
                        debug_img = self._draw_debug_info(base_img, eye_locations)
                        debug_path = get_output_path(base_path).replace("_eye", "_debug")
                        save_image(debug_img, debug_path, None)
                    
                    # Process image
                    result = process_image(base_img, overlay_img, eye_locations)
                    
                    # Save result
                    save_path = save_image(result, None, base_path)
                    
                    self.batch_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    self.batch_failed += 1
            
            # Final status update
            self.status_bar.update_progress(1.0)
            self.status_bar.set_status(
                f"Batch processing complete: {self.batch_processed} processed, "
                f"{self.batch_failed} failed, {self.batch_total} total"
            )
            
            # Show completion message
            self.after(0, lambda: messagebox.showinfo(
                "Batch Processing Complete",
                f"Processed {self.batch_processed} images\n"
                f"Failed: {self.batch_failed} images\n\n"
                f"Output files are in: {file_config.OUTPUT_DIR}"
            ))
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            self.status_bar.set_status(f"Batch processing error: {str(e)}")
        finally:
            self.processing = False
            # Use after to update UI from thread
            self.after(0, lambda: self.process_button.configure(state="normal"))
            self.after(0, lambda: self.batch_button.configure(state="normal"))
            self.after(0, lambda: self.plex_button.configure(state="normal"))

    def _draw_debug_info(self, image: np.ndarray, eye_locations: List[EyeLocation]) -> np.ndarray:
        """
        Draw debug information about face and eye detections.
        
        Args:
            image: Image to draw on
            eye_locations: List of detected eye locations
            
        Returns:
            Image with debug visualizations
        """
        debug_image = image.copy()
        
        for i, eye_loc in enumerate(eye_locations):
            # Draw face rectangle
            face_center_x, face_center_y = eye_loc.face_center
            face_width, face_height = eye_loc.face_size
            face_x = face_center_x - face_width // 2
            face_y = face_center_y - face_height // 2
            
            # Determine color based on confidence (green for high, red for low)
            if eye_loc.confidence >= detection_config.MIN_FACE_CONFIDENCE:
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
            
            # Add face confidence text
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

    def _process_image(self):
        """Handle image processing."""
        if self.processing:
            self.status_bar.set_status("Processing already in progress")
            return
        
        if not self.base_preview.current_image_path:
            self.status_bar.set_status("Please select a base image")
            return
            
        if not self.overlay_preview.current_image_path:
            self.status_bar.set_status("Please select an overlay image")
            return
        
        # Start processing in a separate thread
        self.processing = True
        self.process_button.configure(state="disabled")
        self.batch_button.configure(state="disabled")
        self.status_bar.show_progress()
        
        thread = threading.Thread(target=self._process_image_thread)
        thread.start()

    def _process_image_thread(self):
        """Process the image in a separate thread."""
        try:
            # Load images
            self.status_bar.set_status("Loading images...")
            self.status_bar.update_progress(0.2)
            
            base_img, base_path = load_image(self.base_preview.current_image_path)
            overlay_img, _ = load_image(self.overlay_preview.current_image_path)
            
            # Detect faces
            self.status_bar.set_status("Detecting faces...")
            self.status_bar.update_progress(0.4)
            
            eye_locations = self.face_detector.detect_eyes(base_img)
            
            if not eye_locations:
                self.status_bar.set_status("No faces detected in the image")
                self.processing = False
                self.process_button.configure(state="normal")
                self.batch_button.configure(state="normal")
                self.plex_button.configure(state="normal")
                self.status_bar.hide_progress()
                return
                
            # If debug mode is enabled, save a debug visualization
            if self.debug_mode:
                self.status_bar.set_status("Creating debug visualization...")
                debug_img = self._draw_debug_info(base_img, eye_locations)
                debug_path = get_output_path(base_path).replace("_eye", "_debug")
                save_image(debug_img, debug_path, None)
                self.status_bar.set_status(f"Debug image saved to: {debug_path}")
            
            # Process image
            self.status_bar.set_status("Applying HUGE googly eyes...")
            self.status_bar.update_progress(0.6)
            
            result = process_image(base_img, overlay_img, eye_locations)
            
            # Save result
            self.status_bar.set_status("Saving image...")
            self.status_bar.update_progress(0.8)
            
            # Save to output directory
            save_path = save_image(result, None, base_path)
            
            self.status_bar.set_status(f"Image saved to: {save_path}")
            self.status_bar.update_progress(1.0)
            
        except ImageLoadError as e:
            self.status_bar.set_status(f"Error loading image: {str(e)}")
            logger.error(f"Image load error: {str(e)}")
            
        except ImageSaveError as e:
            self.status_bar.set_status(f"Error saving image: {str(e)}")
            logger.error(f"Image save error: {str(e)}")
            
        except Exception as e:
            self.status_bar.set_status(f"Error processing image: {str(e)}")
            logger.error(f"Processing error: {str(e)}")
            
        finally:
            self.processing = False
            self.process_button.configure(state="normal")
            self.batch_button.configure(state="normal")
            self.plex_button.configure(state="normal")
            self.status_bar.hide_progress()

    def _process_plex_server(self):
        """Open dialog to configure and process images from Plex server via API."""
        if self.processing:
            self.status_bar.set_status("Processing already in progress")
            return
            
        if not self.overlay_preview.current_image_path:
            self.status_bar.set_status("Please select an overlay image first")
            return
            
        # Create dialog window
        dialog = ctk.CTkToplevel(self)
        dialog.title("Plex Server Connection")
        dialog.geometry("630x800")  # Increased size as requested
        dialog.resizable(True, True)  # Make the dialog resizable
        dialog.transient(self)  # Make dialog modal
        dialog.grab_set()
        
        # Server settings frame
        server_frame = ctk.CTkFrame(dialog)
        server_frame.pack(padx=20, pady=20, fill="x")
        
        # Server URL
        ctk.CTkLabel(server_frame, text="Plex Server URL:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        server_url = ctk.CTkEntry(server_frame, width=400)
        server_url.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        server_url.insert(0, plex_config.PLEX_SERVER_URL)
        
        # Auth Token
        ctk.CTkLabel(server_frame, text="Plex Token:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        auth_token = ctk.CTkEntry(server_frame, width=400)
        auth_token.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        auth_token.insert(0, plex_config.PLEX_TOKEN)
        
        # Test connection button
        def test_connection():
            url = server_url.get()
            token = auth_token.get()
            
            if not url or not token:
                connection_status.configure(text="Please enter server URL and token", text_color="red")
                return
                
            try:
                plex_integration = PlexIntegration(self.face_detector)
                plex_server = plex_integration.connect_to_plex(url, token)
                
                if plex_server:
                    # Save settings for next time after successful connection
                    plex_config.PLEX_SERVER_URL = url
                    plex_config.PLEX_TOKEN = token
                    
                    connection_status.configure(
                        text=f"Connected to {plex_server.friendlyName}",
                        text_color="green"
                    )
                    # Get libraries after successful connection
                    libraries = plex_integration.get_libraries(plex_server)
                    
                    # Update library listboxes
                    movie_listbox.delete(0, "end")
                    tv_listbox.delete(0, "end")
                    
                    for lib in libraries:
                        if lib.type == 'movie':
                            movie_listbox.insert("end", lib.title)
                        elif lib.type == 'show':
                            tv_listbox.insert("end", lib.title)
                    
                    # Ensure action buttons are visible
                    process_button.configure(state="normal")
                    action_frame.pack(padx=20, pady=20, fill="x")
                    dialog.update()  # Force update of the dialog
                else:
                    connection_status.configure(
                        text="Failed to connect to Plex server",
                        text_color="red"
                    )
            except Exception as e:
                connection_status.configure(
                    text=f"Error: {str(e)}",
                    text_color="red"
                )
                
        test_button = ctk.CTkButton(
            server_frame, 
            text="Test Connection", 
            command=test_connection,
            fg_color="#e5a00d",  # Yellow color
            text_color="black"
        )
        test_button.grid(row=2, column=0, padx=10, pady=10)
        
        # Connection status
        connection_status = ctk.CTkLabel(server_frame, text="", text_color="gray")
        connection_status.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        # Library selection frame
        library_frame = ctk.CTkFrame(dialog)
        library_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Movie libraries
        ctk.CTkLabel(library_frame, text="Movie Libraries:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        movie_frame = ctk.CTkFrame(library_frame)
        movie_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        movie_listbox = tk.Listbox(movie_frame, selectmode="multiple", bg="#2b2b2b", fg="white", bd=0, height=8)
        movie_listbox.pack(fill="both", expand=True)
        
        # TV libraries
        ctk.CTkLabel(library_frame, text="TV Show Libraries:").grid(row=0, column=1, padx=10, pady=10, sticky="w")
        tv_frame = ctk.CTkFrame(library_frame)
        tv_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        tv_listbox = tk.Listbox(tv_frame, selectmode="multiple", bg="#2b2b2b", fg="white", bd=0, height=8)
        tv_listbox.pack(fill="both", expand=True)
        
        # Configure grid weights for library frame
        library_frame.grid_columnconfigure(0, weight=1)
        library_frame.grid_columnconfigure(1, weight=1)
        library_frame.grid_rowconfigure(1, weight=1)
        
        # Backup/Restore frame
        backup_frame = ctk.CTkFrame(dialog)
        backup_frame.pack(padx=20, pady=10, fill="x")
        
        # Create backup button
        def backup_posters():
            url = server_url.get()
            token = auth_token.get()
            
            # Get selected libraries
            selected_movie_libs = [movie_listbox.get(i) for i in movie_listbox.curselection()]
            selected_tv_libs = [tv_listbox.get(i) for i in tv_listbox.curselection()]
            
            if not url or not token or not (selected_movie_libs or selected_tv_libs):
                messagebox.showwarning(
                    "Selection Required",
                    "Please enter server URL, token, and select at least one library."
                )
                return
                
            # Close dialog
            dialog.grab_release()
            dialog.destroy()
            
            # Process in thread
            self.processing = True
            self.process_button.configure(state="disabled")
            self.batch_button.configure(state="disabled")
            self.plex_button.configure(state="disabled")
            self.plex_server_button.configure(state="disabled")
            self.status_bar.show_progress()
            
            selected_libraries = selected_movie_libs + selected_tv_libs
            
            thread = threading.Thread(
                target=self._backup_plex_posters_thread,
                args=(url, token, selected_libraries)
            )
            thread.start()
            
        backup_button = ctk.CTkButton(
            backup_frame,
            text="BACKUP POSTERS",
            command=backup_posters,
            font=("Helvetica", 14, "bold"),
            fg_color="#e5a00d",  # Yellow color
            text_color="black",
            height=40
        )
        backup_button.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        # Create restore button
        def restore_posters():
            url = server_url.get()
            token = auth_token.get()
            
            # Get selected libraries
            selected_movie_libs = [movie_listbox.get(i) for i in movie_listbox.curselection()]
            selected_tv_libs = [tv_listbox.get(i) for i in tv_listbox.curselection()]
            
            if not url or not token or not (selected_movie_libs or selected_tv_libs):
                messagebox.showwarning(
                    "Selection Required",
                    "Please enter server URL, token, and select at least one library."
                )
                return
                
            # Close dialog
            dialog.grab_release()
            dialog.destroy()
            
            # Process in thread
            self.processing = True
            self.process_button.configure(state="disabled")
            self.batch_button.configure(state="disabled")
            self.plex_button.configure(state="disabled")
            self.plex_server_button.configure(state="disabled")
            self.status_bar.show_progress()
            
            selected_libraries = selected_movie_libs + selected_tv_libs
            
            thread = threading.Thread(
                target=self._restore_plex_posters_thread,
                args=(url, token, selected_libraries)
            )
            thread.start()
            
        restore_button = ctk.CTkButton(
            backup_frame,
            text="RESTORE POSTERS",
            command=restore_posters,
            font=("Helvetica", 14, "bold"),
            fg_color="#e5a00d",  # Yellow color
            text_color="black",
            height=40
        )
        restore_button.pack(side="right", padx=10, pady=10, fill="x", expand=True)
        
        # Backup info label
        backup_info = ctk.CTkLabel(
            backup_frame,
            text=f"Backups stored in: {plex_config.BACKUP_DIR}",
            font=("Helvetica", 10),
            text_color="gray"
        )
        backup_info.pack(side="bottom", padx=10, pady=(0, 5), fill="x")
        
        # Options frame
        options_frame = ctk.CTkFrame(dialog)
        options_frame.pack(padx=20, pady=20, fill="x")
        
        # Debug mode
        debug_var = ctk.BooleanVar(value=self.debug_mode)
        debug_checkbox = ctk.CTkCheckBox(
            options_frame,
            text="Debug Mode (save detection visualization)",
            variable=debug_var
        )
        debug_checkbox.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Movie poster mode
        poster_mode_var = ctk.BooleanVar(value=self.movie_poster_mode)
        poster_mode_checkbox = ctk.CTkCheckBox(
            options_frame,
            text="Movie Poster Mode (better for angled faces)",
            variable=poster_mode_var
        )
        poster_mode_checkbox.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        # Add face confidence threshold slider
        conf_frame = ctk.CTkFrame(options_frame)
        conf_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        conf_frame.grid_columnconfigure(1, weight=1)
        
        conf_label = ctk.CTkLabel(
            conf_frame,
            text="Face Confidence Threshold:",
            anchor="w"
        )
        conf_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Get current confidence threshold
        current_conf = int(detection_config.MIN_FACE_CONFIDENCE * 100)
        
        conf_value_label = ctk.CTkLabel(
            conf_frame,
            text=f"{current_conf}%",
            width=40
        )
        conf_value_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")
        
        def update_conf_value(value):
            conf_percent = int(value)
            conf_value_label.configure(text=f"{conf_percent}%")
        
        conf_slider = ctk.CTkSlider(
            conf_frame,
            from_=0,
            to=100,
            number_of_steps=20,
            command=update_conf_value
        )
        conf_slider.set(current_conf)
        conf_slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # Add eye scale slider
        scale_frame = ctk.CTkFrame(options_frame)
        scale_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        scale_frame.grid_columnconfigure(1, weight=1)
        
        scale_label = ctk.CTkLabel(
            scale_frame,
            text="Eye Size Scale:",
            anchor="w"
        )
        scale_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Convert the decimal scale (0.35) to percentage (35%)
        current_scale = int(detection_config.FACE_BASED_EYE_SCALE * 100)
        
        scale_value_label = ctk.CTkLabel(
            scale_frame,
            text=f"{current_scale}%",
            width=40
        )
        scale_value_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")
        
        def update_scale_value(value):
            scale_percent = int(value)
            scale_value_label.configure(text=f"{scale_percent}%")
        
        scale_slider = ctk.CTkSlider(
            scale_frame,
            from_=10,
            to=70,
            number_of_steps=60,
            command=update_scale_value
        )
        scale_slider.set(current_scale)
        scale_slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # Add auto-backup checkbox
        backup_var = ctk.BooleanVar(value=True)
        backup_checkbox = ctk.CTkCheckBox(
            options_frame,
            text="Automatically backup posters before processing",
            variable=backup_var
        )
        backup_checkbox.grid(row=4, column=0, padx=10, pady=10, sticky="w")
        
        # Action buttons frame
        action_frame = ctk.CTkFrame(dialog)
        action_frame.pack(padx=20, pady=20, fill="x", side="bottom")
        
        # Process button
        def process_plex_server():
            url = server_url.get()
            token = auth_token.get()
            
            # Get selected libraries
            selected_movie_libs = [movie_listbox.get(i) for i in movie_listbox.curselection()]
            selected_tv_libs = [tv_listbox.get(i) for i in tv_listbox.curselection()]
            
            # Get eye scale value (convert from percentage back to decimal)
            eye_scale = scale_slider.get() / 100
            
            # Get face confidence threshold (convert from percentage back to decimal)
            face_confidence = conf_slider.get() / 100
            
            # Get auto-backup setting
            auto_backup = backup_var.get()
            
            # Close dialog
            dialog.grab_release()
            dialog.destroy()
            
            # Start processing in a thread
            if url and token and (selected_movie_libs or selected_tv_libs):
                # Save settings for next time
                plex_config.PLEX_SERVER_URL = url
                plex_config.PLEX_TOKEN = token
                
                # Process in thread
                self.processing = True
                self.process_button.configure(state="disabled")
                self.batch_button.configure(state="disabled")
                self.plex_button.configure(state="disabled")
                self.plex_server_button.configure(state="disabled")
                self.status_bar.show_progress()
                
                selected_libraries = selected_movie_libs + selected_tv_libs
                
                thread = threading.Thread(
                    target=self._process_plex_server_thread,
                    args=(url, token, selected_libraries, debug_var.get(), poster_mode_var.get(), 
                          eye_scale, face_confidence, auto_backup)
                )
                thread.start()
                
        process_button = ctk.CTkButton(
            action_frame,
            text="PROCESS SELECTED LIBRARIES",
            command=process_plex_server,
            font=("Helvetica", 16, "bold"),
            fg_color="#e5a00d",  # Yellow color
            text_color="black",
            height=50
        )
        process_button.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        # Cancel button
        def cancel():
            dialog.grab_release()
            dialog.destroy()
            
        cancel_button = ctk.CTkButton(
            action_frame,
            text="Cancel",
            command=cancel,
            fg_color="#e5a00d",  # Yellow color
            text_color="black",
            height=50
        )
        cancel_button.pack(side="right", padx=10, pady=10)
        
        # Add instruction label
        instruction_label = ctk.CTkLabel(
            dialog, 
            text="After connecting, select libraries and click PROCESS SELECTED LIBRARIES at the bottom",
            font=("Helvetica", 12, "bold"),
            text_color="yellow"
        )
        instruction_label.pack(padx=20, pady=(0, 10), fill="x")

    def _process_plex_server_thread(self, server_url, auth_token, library_names, debug_mode, movie_poster_mode, eye_scale, face_confidence, auto_backup):
        """Process Plex server libraries in a background thread."""
        try:
            self.status_bar.set_status("Connecting to Plex server...")
            self.status_bar.update_progress(0.1)
            
            # Store original settings to restore later
            original_scale = detection_config.FACE_BASED_EYE_SCALE
            original_confidence = detection_config.MIN_FACE_CONFIDENCE
            
            # Apply settings for this batch processing
            detection_config.FACE_BASED_EYE_SCALE = eye_scale
            detection_config.MIN_FACE_CONFIDENCE = face_confidence
            
            # Initialize Plex integration
            plex_integration = PlexIntegration(self.face_detector)
            
            # Connect to Plex server
            plex_server = plex_integration.connect_to_plex(server_url, auth_token)
            if not plex_server:
                self.status_bar.set_status("Failed to connect to Plex server")
                # Restore original settings
                detection_config.FACE_BASED_EYE_SCALE = original_scale
                detection_config.MIN_FACE_CONFIDENCE = original_confidence
                return
                
            self.status_bar.set_status(f"Connected to Plex server: {plex_server.friendlyName}")
            self.status_bar.update_progress(0.2)
            
            # Create a progress update callback function
            def update_progress(current, total, item_name=""):
                progress = 0.2 + (0.75 * (current / total if total > 0 else 0))
                self.status_bar.update_progress(progress)
                if item_name:
                    self.status_bar.set_status(f"Processing {current}/{total}: {item_name}")
                else:
                    self.status_bar.set_status(f"Processing {current}/{total} items...")
                    
            # Load overlay image
            overlay_img, _ = load_image(self.overlay_preview.current_image_path)
            
            # Process libraries 
            self.status_bar.set_status(f"Counting items in {len(library_names)} libraries...")
            
            # Add a callback to update progress
            plex_integration.set_progress_callback(update_progress)
            
            # Process the libraries
            stats = plex_integration.process_library_items(
                plex_server,
                library_names,
                overlay_img,
                debug_mode,
                movie_poster_mode,
                backup_before_processing=auto_backup
            )
            
            # Restore original settings
            detection_config.FACE_BASED_EYE_SCALE = original_scale
            detection_config.MIN_FACE_CONFIDENCE = original_confidence
            
            # Show completion message
            self.status_bar.update_progress(1.0)
            self.status_bar.set_status(
                f"Plex server processing complete: {stats['processed']} processed, "
                f"{stats['failed']} failed, {stats['no_face']} with no faces, {stats['total']} total"
            )
            
            # Cleanup temporary files
            plex_integration.cleanup()
            
            # Show completion message
            self.after(0, lambda: messagebox.showinfo(
                "Plex Server Processing Complete",
                f"Processed: {stats['processed']} images\n"
                f"Failed: {stats['failed']} images\n"
                f"No faces detected: {stats['no_face']} images\n"
                f"Total items: {stats['total']}"
            ))
            
        except Exception as e:
            logger.error(f"Plex server processing error: {str(e)}")
            self.status_bar.set_status(f"Plex server processing error: {str(e)}")
            
            # Show error message
            self.after(0, lambda: messagebox.showerror(
                "Plex Server Processing Error",
                f"An error occurred: {str(e)}"
            ))
            
        finally:
            self.processing = False
            # Use after to update UI from thread
            self.after(0, lambda: self.process_button.configure(state="normal"))
            self.after(0, lambda: self.batch_button.configure(state="normal"))
            self.after(0, lambda: self.plex_button.configure(state="normal"))
            self.after(0, lambda: self.plex_server_button.configure(state="normal"))
            self.after(0, lambda: self.status_bar.hide_progress())

    def _backup_plex_posters_thread(self, url, token, library_names):
        """Thread function to backup Plex posters."""
        try:
            self.status_bar.set_status("Connecting to Plex server...")
            self.status_bar.update_progress(0.1)
            
            # Initialize Plex integration
            plex_integration = PlexIntegration(self.face_detector)
            
            # Connect to Plex server
            plex_server = plex_integration.connect_to_plex(url, token)
            if not plex_server:
                self.status_bar.set_status("Failed to connect to Plex server")
                return
                
            self.status_bar.set_status(f"Connected to Plex server: {plex_server.friendlyName}")
            self.status_bar.update_progress(0.2)
            
            # Create a progress update callback function
            def update_progress(current, total, item_name=""):
                progress = 0.2 + (0.75 * (current / total if total > 0 else 0))
                self.status_bar.update_progress(progress)
                if item_name:
                    self.status_bar.set_status(f"Backing up {current}/{total}: {item_name}")
                else:
                    self.status_bar.set_status(f"Backing up {current}/{total} items...")
            
            # Set progress callback
            plex_integration.set_progress_callback(update_progress)
            
            # Process the libraries
            stats = plex_integration.backup_posters(
                plex_server,
                library_names
            )
            
            # Show completion message
            self.status_bar.update_progress(1.0)
            self.status_bar.set_status(
                f"Plex server backup complete: {stats['backed_up']} backed up, "
                f"{stats['failed']} failed, {stats['total']} total"
            )
            
            # Show completion dialog
            self.after(0, lambda: messagebox.showinfo(
                "Plex Poster Backup Complete",
                f"Successfully backed up {stats['backed_up']} posters.\n"
                f"Failed: {stats['failed']} posters\n"
                f"Total items: {stats['total']}\n\n"
                f"Backups stored in: {plex_config.BACKUP_DIR}"
            ))
            
        except Exception as e:
            logger.error(f"Plex server backup error: {str(e)}")
            self.status_bar.set_status(f"Plex server backup error: {str(e)}")
            
            # Show error message
            self.after(0, lambda: messagebox.showerror(
                "Plex Server Backup Error",
                f"An error occurred: {str(e)}"
            ))
            
        finally:
            self.processing = False
            # Use after to update UI from thread
            self.after(0, lambda: self.process_button.configure(state="normal"))
            self.after(0, lambda: self.batch_button.configure(state="normal"))
            self.after(0, lambda: self.plex_button.configure(state="normal"))
            self.after(0, lambda: self.plex_server_button.configure(state="normal"))
            self.after(0, lambda: self.status_bar.hide_progress())

    def _restore_plex_posters_thread(self, url, token, library_names):
        """Thread function to restore Plex posters."""
        try:
            self.status_bar.set_status("Connecting to Plex server...")
            self.status_bar.update_progress(0.1)
            
            # Initialize Plex integration
            plex_integration = PlexIntegration(self.face_detector)
            
            # Connect to Plex server
            plex_server = plex_integration.connect_to_plex(url, token)
            if not plex_server:
                self.status_bar.set_status("Failed to connect to Plex server")
                return
                
            self.status_bar.set_status(f"Connected to Plex server: {plex_server.friendlyName}")
            self.status_bar.update_progress(0.2)
            
            # Create a progress update callback function
            def update_progress(current, total, item_name=""):
                progress = 0.2 + (0.75 * (current / total if total > 0 else 0))
                self.status_bar.update_progress(progress)
                if item_name:
                    self.status_bar.set_status(f"Restoring {current}/{total}: {item_name}")
                else:
                    self.status_bar.set_status(f"Restoring {current}/{total} items...")
            
            # Set progress callback
            plex_integration.set_progress_callback(update_progress)
            
            # Process the libraries
            stats = plex_integration.restore_posters(
                plex_server,
                library_names
            )
            
            # Show completion message
            self.status_bar.update_progress(1.0)
            self.status_bar.set_status(
                f"Plex server restore complete: {stats['restored']} restored, "
                f"{stats['failed']} failed, {stats['no_backup']} without backups"
            )
            
            # Show completion dialog
            self.after(0, lambda: messagebox.showinfo(
                "Plex Poster Restore Complete",
                f"Successfully restored {stats['restored']} posters.\n"
                f"Failed: {stats['failed']} posters\n"
                f"Items without backups: {stats['no_backup']}\n"
                f"Total items checked: {stats['total']}"
            ))
            
        except Exception as e:
            logger.error(f"Plex server restore error: {str(e)}")
            self.status_bar.set_status(f"Plex server restore error: {str(e)}")
            
            # Show error message
            self.after(0, lambda: messagebox.showerror(
                "Plex Server Restore Error",
                f"An error occurred: {str(e)}"
            ))
            
        finally:
            self.processing = False
            # Use after to update UI from thread
            self.after(0, lambda: self.process_button.configure(state="normal"))
            self.after(0, lambda: self.batch_button.configure(state="normal"))
            self.after(0, lambda: self.plex_button.configure(state="normal"))
            self.after(0, lambda: self.plex_server_button.configure(state="normal"))
            self.after(0, lambda: self.status_bar.hide_progress())

    def run(self):
        """Start the application main loop."""
        self.mainloop() 