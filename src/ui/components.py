"""
Reusable UI components for the Eye Image Overlay Application.
"""
from typing import Optional, Callable, Tuple
import customtkinter as ctk
from PIL import Image
import logging

from ..utils.config import ui_config

logger = logging.getLogger(__name__)

class ImagePreview(ctk.CTkFrame):
    """A frame for displaying image previews with a label and button."""
    
    def __init__(
        self,
        master: ctk.CTk,
        title: str,
        button_text: str,
        command: Callable,
        **kwargs
    ):
        """
        Initialize the image preview frame.

        Args:
            master: Parent widget
            title: Title text for the frame
            button_text: Text for the select button
            command: Callback function for the button
        """
        super().__init__(master, **kwargs)
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Create widgets
        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=("Helvetica", 14, "bold")
        )
        self.title_label.grid(row=0, column=0, padx=10, pady=(10, 5))
        
        # Create a frame for the preview image with a border
        self.preview_frame = ctk.CTkFrame(
            self,
            width=ui_config.PREVIEW_SIZE[0],
            height=ui_config.PREVIEW_SIZE[1],
            fg_color=("gray90", "gray20")  # Light and dark mode colors
        )
        self.preview_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.preview_frame.grid_propagate(False)  # Prevent frame from resizing to content
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)
        
        # Create the preview label inside the frame
        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="No image selected",
            corner_radius=0,
            fg_color="transparent"
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        
        # Create a button with better styling
        self.select_button = ctk.CTkButton(
            self,
            text=button_text,
            command=command,
            height=32,
            corner_radius=6,
            font=("Helvetica", 12),
            fg_color="#e5a00d",  # Yellow color
            text_color="black"
        )
        self.select_button.grid(row=2, column=0, padx=10, pady=10)
        
        # Store the current preview image and path
        self.current_preview: Optional[ctk.CTkImage] = None
        self.current_image_path: Optional[str] = None

    def update_preview(self, image: Image.Image, image_path: str):
        """
        Update the preview with a new image.

        Args:
            image: PIL Image to display
            image_path: Path of the selected image
        """
        try:
            # Create a copy of the image to avoid modifying the original
            img_copy = image.copy()
            
            # Calculate aspect ratio
            width, height = img_copy.size
            aspect_ratio = width / height
            
            # Calculate dimensions that fit within preview size while maintaining aspect ratio
            preview_width, preview_height = ui_config.PREVIEW_SIZE
            
            if aspect_ratio > 1:  # Wider than tall
                new_width = min(width, preview_width)
                new_height = int(new_width / aspect_ratio)
            else:  # Taller than wide
                new_height = min(height, preview_height)
                new_width = int(new_height * aspect_ratio)
            
            # Create CTkImage with proper dimensions for high DPI support
            self.current_preview = ctk.CTkImage(
                light_image=img_copy,
                dark_image=img_copy,
                size=(new_width, new_height)
            )
            self.current_image_path = image_path
            
            # Update label
            self.preview_label.configure(image=self.current_preview, text="")
            
        except Exception as e:
            logger.error(f"Failed to update preview: {str(e)}")
            self.preview_label.configure(text="Failed to load preview")
            self.current_preview = None
            self.current_image_path = None

    def clear_preview(self):
        """Clear the current preview."""
        self.preview_label.configure(text="No image selected", image=None)
        self.current_preview = None
        self.current_image_path = None

class StatusBar(ctk.CTkFrame):
    """Status bar for displaying application status and progress."""
    
    def __init__(self, master: ctk.CTk, **kwargs):
        """Initialize the status bar."""
        super().__init__(master, height=60, fg_color=("gray85", "gray25"), **kwargs)
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        
        # Create status label
        self.status_label = ctk.CTkLabel(
            self,
            text="Ready",
            anchor="w",
            padx=10,
            font=("Helvetica", 12)
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        # Create progress bar with better styling
        self.progress_bar = ctk.CTkProgressBar(
            self,
            height=12,
            corner_radius=2,
            border_width=0
        )
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.progress_bar.set(0)
        self.progress_bar.grid_remove()  # Hide initially

    def set_status(self, text: str):
        """Update the status text."""
        self.status_label.configure(text=text)

    def show_progress(self):
        """Show and reset the progress bar."""
        self.progress_bar.grid()
        self.progress_bar.set(0)

    def hide_progress(self):
        """Hide the progress bar."""
        self.progress_bar.grid_remove()

    def update_progress(self, value: float):
        """Update the progress bar value (0-1)."""
        self.progress_bar.set(value)

class SaveLocationFrame(ctk.CTkFrame):
    """Frame for selecting save location."""
    
    def __init__(
        self,
        master: ctk.CTk,
        command: Callable,
        **kwargs
    ):
        """
        Initialize the save location frame.

        Args:
            master: Parent widget
            command: Callback function for the browse button
        """
        super().__init__(master, **kwargs)
        
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        
        # Create widgets
        self.label = ctk.CTkLabel(
            self,
            text="Save Location (optional):",
            anchor="w"
        )
        self.label.grid(row=0, column=0, padx=5, pady=5)
        
        self.path_entry = ctk.CTkEntry(self)
        self.path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.browse_button = ctk.CTkButton(
            self,
            text="Browse",
            command=command,
            width=100,
            fg_color="#e5a00d",  # Yellow color
            text_color="black"
        )
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

    def get_save_path(self) -> str:
        """Get the current save path."""
        return self.path_entry.get()

    def set_save_path(self, path: str):
        """Set the save path."""
        self.path_entry.delete(0, "end")
        self.path_entry.insert(0, path) 