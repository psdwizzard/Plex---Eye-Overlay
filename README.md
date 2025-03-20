# Eye Image Overlay Application

An application that detects eyes in images and overlays googly eyes on them for humorous effect.

![Googly Eyes Example](docs/examples.jpg)

## Features

- User-friendly interface with image previews and progress reporting
- Multiple face detection with eye landmark detection
- Googly eye overlays that scale proportionally to face size
- Support for transparent PNG overlays
- Proper handling of image orientation from EXIF data
- Debug mode to visualize face and eye detections
- Movie poster mode for improved detection of faces at odd angles
- Adjustable confidence threshold slider for fine-tuning detection sensitivity
- Batch processing for adding googly eyes to entire folders of images
- Plex Library mode to recursively process images in all subfolders and overwrite originals
- Special support for Plex metadata files (extension-less files in "Uploads/posters" and files like "tv.plex.agents.*")
- Direct Plex API integration for processing posters from your Plex Media Server

## Installation

1. Clone the repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python src/main.py
   ```

## Usage

1. Launch the application
2. Load a base image by clicking "Select Base Image"
3. (Optional) Select a custom eye overlay image by clicking "Select Eye Image"
4. Configure settings as needed:
   - (Optional) Enable "Debug Mode" to visualize face and eye detections
   - (Optional) Enable "Movie Poster Mode" for better detection on movie posters or faces at odd angles
   - (Optional) Adjust the confidence threshold slider - lower values (30-40%) work well for movie posters
5. Process images:
   - Click "Add HUGE Googly Eyes" to process the current image
   - Click "Batch Process Folder" to apply googly eyes to all images in a folder
   - Click "Process Plex Library" to recursively process all images in a folder and its subfolders, overwriting the original files
   - Click "Process Plex Server" to connect directly to your Plex Media Server via API and process posters
6. The processed images will be saved in the output directory (except for Plex modes, which overwrite originals)

## How It Works

- **Image Loading**: The application loads images and automatically corrects orientation using EXIF data to ensure proper display regardless of how the photo was taken.

- **Face Detection**: Using MediaPipe Face Detection, the application identifies faces in the image and finds the precise location of eyes using facial landmarks.

- **Confidence Threshold**: The confidence slider lets you adjust how certain the app needs to be before accepting a face detection. Lower values (30%) will detect more faces but may cause "floating eyes" in complex images. Higher values (70%+) are more conservative but may miss some faces.

- **Eye Scaling**: Googly eyes are automatically sized proportionally to the detected face dimensions, ensuring they look appropriate for each person's face size.

- **Eye Placement**: The application precisely places the googly eyes over the detected eye positions, maintaining the natural look of the face.

- **Transparency Support**: The application supports PNG files with transparency, ensuring that only the eye part of the overlay appears without any background.

- **Debug Mode**: When enabled, the application generates a visualization showing detected faces (green for high-confidence, red for low-confidence detections), eye positions, and their sizes to help diagnose detection issues.

- **Movie Poster Mode**: Specialized detection optimized for movie posters and faces at odd angles. Uses additional detection techniques like profile face detection and enhanced contrast to find faces that standard detection might miss.

- **Batch Processing**: Process multiple images at once by selecting a folder. The application will apply googly eyes to all supported image files in the folder, saving the results to the output directory.

- **Plex Library Processing**: Recursively processes all images in a folder and all its subfolders. Unlike normal batch processing, this option overwrites the original files rather than saving to the output directory, making it ideal for updating Plex media libraries.

  - **Plex Metadata Support**: The app can detect and process Plex's metadata image files, including:
    - Extension-less files in paths containing "Uploads/posters"
    - Files with names starting with "tv.plex.agents.", "com.plexapp.agents.", etc.
    - Any extension-less file that contains valid image data (JPEG or PNG)

- **Plex API Integration**: Connect directly to your Plex Media Server using the official Plex API to:
  - Download posters for movies, TV shows, and seasons
  - Apply googly eyes to the posters
  - Upload the modified posters back to your Plex server
  - Process multiple libraries at once

- **Output**: For standard processing, images are saved with an "_eye" suffix appended to the original filename. Debug images (when debug mode is enabled) are saved with a "_debug" suffix. In Plex modes, original images are overwritten.

## Plex Media Server Integration

The application offers two methods for integrating with Plex:

### 1. Filesystem-based Integration (Process Plex Library)

This approach processes files directly on the filesystem:

1. It can detect and process extension-less image files in Plex's metadata structure
2. Supports multiple Plex file patterns:
   - Files in paths containing "Uploads/posters"
   - Files with names starting with "tv.plex.agents." (TV show artwork)
   - Files with names starting with "com.plexapp.agents." (movie artwork)
   - Other Plex agent-specific metadata files
3. Identifies JPEG and PNG formats by examining the file's binary signature
4. Preserves the original filenames and formats when saving
5. When using debug mode with Plex files, debug images are saved to the output directory with a special naming convention

To use this feature:
1. Select the "Process Plex Library" button
2. Browse to your Plex Media Server's metadata directory (typically in "Plex Media Server/Metadata/")
3. The app will recursively scan for all regular images and special Plex metadata files
4. Confirm the prompt to process and overwrite files

### 2. API-based Integration (Process Plex Server)

This approach uses the Plex API to directly communicate with your Plex server:

1. Connects to your Plex Media Server using the Plex API
2. Downloads posters for movies, TV shows, and seasons
3. Processes the posters to add googly eyes
4. Uploads the modified posters back to your Plex server

Benefits of the API approach:
- No need to locate and modify files on disk
- Works with any Plex server you have access to, including remote servers
- Automatically handles all poster types and formats
- Processes all items in selected libraries with one click

To use this feature:
1. Click the "Process Plex Server" button
2. Enter your Plex server URL (e.g., http://localhost:32400) and authentication token
3. Click "Test Connection" to verify connectivity and fetch libraries
4. Select the libraries you want to process
5. Click "Process Selected Libraries" to begin

To find your Plex authentication token, follow [this guide](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/).

## Configuration

The application behavior can be customized by modifying the configuration files in the `src/utils` directory:

- `config.py`: Contains settings for detection thresholds, image processing parameters, UI preferences, and Plex integration settings.

## Development

The project is structured with modularity in mind:

- `src/processors`: Contains image processing logic
- `src/ui`: Contains user interface code
- `src/utils`: Contains utility functions and configuration

## License

[MIT License](LICENSE)

## Troubleshooting

- **No faces detected**: Try using a clearer image with well-lit, front-facing subjects. Enable debug mode to visualize detection results. For movie posters or faces at angles, enable Movie Poster Mode and lower the confidence threshold to 30-40%.

- **Floating eyes**: If eyes appear in wrong places, enable debug mode to see detection quality. Low-confidence detections (red boxes) may cause floating eyes. Try increasing the confidence threshold if this happens.

- **Poor overlay alignment**: The application works best with front-facing, well-lit faces. Profile shots may have less accurate eye placement. Movie Poster Mode can help with these cases.

- **Issues with movie posters**: For movie posters with faces that are hard to detect (due to angles, hats, stylized artwork), enable Movie Poster Mode and lower the confidence threshold to 30%.

- **Plex metadata files not recognized**: Ensure you're selecting a directory that contains Plex metadata. The app recognizes:
  - Extension-less files in paths containing "Uploads/posters"
  - Files with names starting with "tv.plex.agents.", "com.plexapp.agents.", etc.
  - If still having issues, enable debug mode to see if files are being detected but failing face detection

- **Plex API connection issues**: 
  - Verify your Plex server URL includes the port (usually 32400)
  - Confirm your authentication token is correct and not expired
  - Ensure your Plex server is running and accessible from your computer
  - Check that your user account has permission to modify the libraries you're trying to access

- **UI display issues**: If the UI appears distorted or has scaling issues, try adjusting the window size in `src/utils/config.py` by modifying the `WINDOW_SIZE` parameter. 
