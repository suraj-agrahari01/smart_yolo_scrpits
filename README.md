# YOLOv5 Annotation Tool

This Python script is designed to assist in annotating images using YOLOv5 object detection. It allows users to visualize object detections, add or edit annotations, and save annotated images and labels.

## Features

-   Load images from a specified directory.
-   Display detected objects and their class labels.
-   Add, edit, or remove annotations manually.
-   Save annotated images and corresponding labels.
-   Zoom in/out, rotate images for detailed annotation.
-   Navigate through images using keyboard shortcuts.

## Dependencies

-   Python 3.x
-   OpenCV (`cv2`)
-   PyTorch
-   NumPy
-   YOLOv5 model from ultralytics

## Installation

1. Install Python 3.x if not already installed.
2. Install required dependencies using pip:
3. Install YOLOv5 model from ultralytics:

## Usage

1. Clone or download the repository to your local machine.
2. Navigate to the directory containing the script.
3. Ensure your images are stored in a directory named `images` within the main directory.
4. Run the script using Python:
5. Follow on-screen instructions to annotate images.
6. Use keyboard shortcuts to navigate, zoom, rotate, and save annotations.

## Keyboard Shortcuts

-   `ESC`: Exit the program.
-   `n`: Move to the next image.
-   `p`: Move to the previous image.
-   `s`: Save the annotated image.
-   `y`: Save the labels.
-   `+` or `=`: Zoom in.
-   `-` or `_`: Zoom out.
-   `r`: Rotate the image.
-   `h`: Display help menu.

## Contributing

Contributions are welcome! Feel free to submit bug reports, feature requests, or pull requests to improve this tool.

## License

This project is licensed under the MIT License.
