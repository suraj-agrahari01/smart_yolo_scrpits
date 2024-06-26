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

# YOLOv5 Elephant Detection Tool

This Python script is designed to detect elephants in videos or images using YOLOv5 object detection. It provides options to process both videos and image folders, visualize detected elephants, and save annotated frames and labels.

## Features

-   Detect elephants in videos or image folders.
-   Set confidence threshold for detection.
-   Process videos up to a desired length.
-   Option to crop and save detected elephants as separate images.
-   Save annotated frames and labels in YOLO format.

## Dependencies

-   Python 3.x
-   OpenCV (`cv2`)
-   PyTorch
-   YOLOv5 model from ultralytics

## Installation

1. Install Python 3.x if not already installed.
2. Install required dependencies using pip:
3. Install YOLOv5 model from ultralytics:

## Usage

1. Set the input path to your video or image folder in the script.
2. Set the desired confidence threshold and processing options.
3. Run the script using Python:
4. Check the output directory for annotated frames and labels.

## Contributing

Contributions are welcome! Feel free to submit bug reports, feature requests, or pull requests to improve this tool.

## License

This project is licensed under the MIT License.
