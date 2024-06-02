import torch
import cv2
import os
from datetime import datetime

# Function to convert bounding box to YOLO format
def convert_to_yolo_format(x1, y1, x2, y2, image_width, image_height, label):
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = width / image_width
    height_norm = height / image_height
    yolo_format = f"{label} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"
    return yolo_format

# Function to write data to a file


def write_to_file(filename, data):
    with open(filename, 'a') as file:
        file.write(data)

# Function to process a video


def process_video(input_video_path, output_video_path, confidence_threshold, desired_length_seconds, model, crop_images):
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("the output dir ", output_dir)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Current frame per second is ", fps)

    # Calculate the maximum number of frames
    max_frames = int(desired_length_seconds * fps)
    frame_count = 0

    # Create folder to save frames with detected elephants
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"detections_{current_datetime}"
    folder_path = os.path.join(output_dir, folder_name)
    cropped_folder_path = os.path.join(folder_path, 'cropped')
    if crop_images:
        os.makedirs(cropped_folder_path, exist_ok=True)

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame, model, confidence_threshold,
                      folder_path, cropped_folder_path, frame_count, width, height, crop_images)

        out.write(frame)
        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    print(
        f"Processed {frame_count} frames and saved video as {output_video_path}")
    print(f"Frames containing elephants are saved in folder: {folder_path}")

# Function to process a folder of images


def process_image_folder(input_folder_path, output_folder_path, confidence_threshold, model, crop_images):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)

    # Create folder to save frames with detected elephants
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"detections_{current_datetime}"
    folder_path = os.path.join(output_folder_path, folder_name)
    cropped_folder_path = os.path.join(folder_path, 'cropped')
    if crop_images:
        os.makedirs(cropped_folder_path, exist_ok=True)

    image_files = [f for f in os.listdir(
        input_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for frame_count, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder_path, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        height, width = frame.shape[:2]

        process_frame(frame, model, confidence_threshold,
                      folder_path, cropped_folder_path, frame_count, width, height, crop_images)

        # Save the processed frame
        output_image_path = os.path.join(output_folder_path, image_file)
        cv2.imwrite(output_image_path, frame)

    print(
        f"Processed {len(image_files)} images and saved them in {output_folder_path}")
    print(f"Frames containing elephants are saved in folder: {folder_path}")

# Function to process each frame


def process_frame(frame, model, confidence_threshold, folder_path, cropped_folder_path, frame_count, width, height, crop_images):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    bounding_boxes = []

    detections = results.pred[0]
    for det in detections:
        label = model.names[int(det[-1])]
        confidence = det[4].item()
        bbox = det[:4].tolist()
        if label == 'elephant' and confidence > confidence_threshold:
            x1, y1, x2, y2 = map(int, bbox)
            bounding_boxes.append((x1, y1, x2, y2))

    if bounding_boxes:
        save_frame_and_labels(frame, bounding_boxes,
                              folder_path, cropped_folder_path, frame_count, width, height, crop_images)

# Function to save frames and labels


def save_frame_and_labels(frame, bounding_boxes, folder_path, cropped_folder_path, frame_count, width, height, crop_images):
    image_path = os.path.join(folder_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    original_frame_filename = os.path.join(
        image_path, f"elephant_frame_{frame_count}_original.jpg")
    cv2.imwrite(original_frame_filename, frame)
    print(original_frame_filename)

    crop_counter = 0

    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, 'elephant', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if crop_images:
            cropped_image = frame[y1:y2, x1:x2]
            cropped_image_filename = os.path.join(
                cropped_folder_path, f"elephant_frame_{frame_count}_{crop_counter}_cropped.jpg")
            cv2.imwrite(cropped_image_filename, cropped_image)
            print(cropped_image_filename)
            crop_counter = crop_counter + 1

    label_path = os.path.join(folder_path, 'labels')
    os.makedirs(label_path, exist_ok=True)
    label_filename = os.path.join(
        label_path, f"elephant_frame_{frame_count}_original.txt")

    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        yolo_data = convert_to_yolo_format(
            x1, y1, x2, y2, width, height, 'elephant')
        write_to_file(label_filename, yolo_data)
    print(label_filename)


# Main execution
if __name__ == "__main__":
    # Replace with the path to your input video or image folder
    input_path = 'e.mp4'
    confidence_threshold = 0.25
    desired_length_seconds = 10

    crop_images = True  # Set to True to enable cropping

    # Load model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    if os.path.isfile(input_path):
        output_path = 'output_video/output_video.mp4'
        process_video(input_path, output_path,
                      confidence_threshold, desired_length_seconds, model, crop_images)
    elif os.path.isdir(input_path):
        output_path = 'output_video'
        process_image_folder(input_path, output_path,
                             confidence_threshold, model, crop_images)
    else:
        print("Error: The input path is neither a file nor a directory.")
