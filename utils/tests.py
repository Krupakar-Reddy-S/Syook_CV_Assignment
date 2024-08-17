import os
import sys
import cv2
import yaml
import shutil
import random

# add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# color codes for different classes
from utils.vars import class_colors

# convert normalized box coordinates to pixel coordinates
def convert_box(box, width, height):
    x_center, y_center, w, h = box
    xmin = int((x_center - w / 2) * width)
    ymin = int((y_center - h / 2) * height)
    xmax = int((x_center + w / 2) * width)
    ymax = int((y_center + h / 2) * height)
    return xmin, ymin, xmax, ymax

# draw bounding boxes on an image with class names
def draw_bounding_boxes(image_path, label_path, class_names, class_colors):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        box = list(map(float, parts[1:]))
        xmin, ymin, xmax, ymax = convert_box(box, width, height)

        color = class_colors.get(class_names[class_id], (0, 0, 0))  # default to black if class color is not found
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, class_names[class_id], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image

# load class names from data.yml file
def load_class_names(data_yml_path):
    with open(data_yml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

# draw bounding boxes on random images from a dataset
def process_images(images_dir, labels_dir, output_dir, class_names, class_colors, num_images=10):
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random_images = random.sample(image_files, num_images)

    # if the output directory exists, remove it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for image_file in random_images:
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')

        output_image = draw_bounding_boxes(image_path, label_path, class_names, class_colors)
        output_image_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_image_path, output_image)
