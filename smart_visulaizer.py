import cv2
import os
import torch
import numpy as np
from pathlib import Path

# Main directory containing subfolders for images and labels
main_dir = 'detections_20240602_101943'
images_dir = os.path.join(main_dir, 'images')
labels_dir = os.path.join(main_dir, 'labels')

# Ensure the labels directory exists
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

# Directory to save annotated images
output_dir = os.path.join(main_dir, 'annotated_images')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load YOLOv5 model
# You can choose other models like yolov5m, yolov5l, yolov5x
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Get list of all files in the images directory
image_filenames = [f for f in os.listdir(
    images_dir) if f.endswith('.jpg') or f.endswith('.png')]

index = 0
scale = 1.0
rotation_angle = 0


def draw_annotations(img, labels, model):
    height, width, _ = img.shape

    for label in labels:
        try:
            parts = label.split()
            cls_name = parts[0]  # Treat the first part as class name string
            x_center, y_center, w, h = map(float, parts[1:])
            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        except ValueError as e:
            print(f"Skipping invalid label: {label}. Error: {e}")


while index < len(image_filenames):
    filename = image_filenames[index]
    img_path = os.path.join(images_dir, filename)
    print("Image Name:", filename)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (500, 500))
    height, width, channels = img.shape

    # Find the corresponding label file in the labels directory
    label_filename = os.path.splitext(filename)[0] + '.txt'
    text_path = os.path.join(labels_dir, label_filename)

    labels = []
    if os.path.exists(text_path):
        with open(text_path, 'r') as fl:
            labels = fl.readlines()

    if not labels:
        results = model(img)
        detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, cls

        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            cls_id = int(cls)
            cls_name = model.names[cls_id]
            labels.append(f"{cls_name} {x_center} {y_center} {w} {h}\n")

    draw_annotations(img, labels, model)

    # Display additional information
    cv2.putText(img, f"Image {index + 1}/{len(image_filenames)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the image
    cv2.imshow("Image", img)

    while True:
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            index = len(image_filenames)  # Exit outer loop
            break
        elif key == ord('n'):  # 'n' key for next image
            index += 1
            break
        elif key == ord('p'):  # 'p' key for previous image
            index -= 1
            break
        elif key == ord('s'):  # 's' key to save the image
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, img)
            print(f"Image saved as {save_path}")
        elif key == ord('y'):  # 'y' key to save the labels
            with open(text_path, 'w') as fl:
                fl.writelines(labels)
            print(f"Labels saved as {text_path}")
        elif key == ord('+') or key == ord('='):  # '+' key to zoom in
            scale *= 1.2
            img_resized = cv2.resize(
                img, (int(width * scale), int(height * scale)))
            cv2.imshow("Image", img_resized)
        elif key == ord('-') or key == ord('_'):  # '-' key to zoom out
            scale /= 1.2
            img_resized = cv2.resize(
                img, (int(width * scale), int(height * scale)))
            cv2.imshow("Image", img_resized)
        elif key == ord('r'):  # 'r' key to rotate the image
            rotation_angle = (rotation_angle + 90) % 360
            M = cv2.getRotationMatrix2D(
                (width / 2, height / 2), rotation_angle, 1)
            img_rotated = cv2.warpAffine(img, M, (width, height))
            cv2.imshow("Image", img_rotated)
        elif key == ord('h'):  # 'h' key to display help
            help_text = """
            ESC: Exit
            n: Next Image
            p: Previous Image
            s: Save Image
            y: Save Labels
            +: Zoom In
            -: Zoom Out
            r: Rotate Image
            h: Display Help
            """
            print(help_text)

    cv2.destroyAllWindows()

print("Program stopped!")
