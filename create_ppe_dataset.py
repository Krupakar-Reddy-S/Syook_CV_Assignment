import os
import cv2
import json
import argparse
import numpy as np

from utils.utils import split_dataset
from utils.vars import classes, ppe_max_count

def crop_person_images_and_filter_annotations(images_folder, labels_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    temp_images_folder = os.path.join(output_folder, 'temp_images')
    temp_labels_folder = os.path.join(output_folder, 'temp_labels')
    os.makedirs(temp_images_folder, exist_ok=True)
    os.makedirs(temp_labels_folder, exist_ok=True)

    cropping_info = {}  # store cropping information

    for image_file in os.listdir(images_folder):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(images_folder, image_file)
            label_path = os.path.join(labels_folder, os.path.splitext(image_file)[0] + '.txt')
            
            with open(label_path, 'r') as infile:
                person_count = 0
                for line in infile:
                    if line.startswith("0 "):  # search for 'person' class
                        person_count += 1
                        cropped_image_name = f"{os.path.splitext(image_file)[0]}_{person_count}.jpg"
                        cropped_label_name = f"{os.path.splitext(image_file)[0]}_{person_count}.txt"
                        cropped_image_path = os.path.join(temp_images_folder, cropped_image_name)
                        cropped_label_path = os.path.join(temp_labels_folder, cropped_label_name)
                        
                        # crop the image and save it
                        xmin, ymin, xmax, ymax = crop_image_and_save(image_path, line, cropped_image_path)
                        
                        cropping_info[cropped_image_name] = {
                            "original_image": image_file,
                            "crop_coords": [xmin, ymin, xmax, ymax]
                        }
                        
                        # filter and save annotations
                        filter_and_save_annotations(label_path, cropped_label_path, person_index=person_count)

    # save cropping information
    with open(os.path.join(output_folder, 'cropping_info.json'), 'w') as f:
        json.dump(cropping_info, f)

    return temp_images_folder, temp_labels_folder

def crop_image_and_save(image_path, bbox_line, output_image_path):
    # parse the bounding box information
    bbox_parts = bbox_line.strip().split()
    x_center, y_center, w, h = map(float, bbox_parts[1:])

    # read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # convert normalized box coordinates to pixel coordinates
    xmin = int((x_center - w / 2) * width)
    ymin = int((y_center - h / 2) * height)
    xmax = int((x_center + w / 2) * width)
    ymax = int((y_center + h / 2) * height)

    # ensure the coordinates are within the image boundaries
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)

    # crop the image and save it
    cropped_image = image[ymin:ymax, xmin:xmax]

    cv2.imwrite(output_image_path, cropped_image)

    # return the coordinates of the cropped image
    return xmin, ymin, xmax, ymax

def filter_and_save_annotations(label_path, output_label_path, person_index):
    with open(label_path, 'r') as infile, open(output_label_path, 'w') as outfile:
        lines = infile.readlines()
        person_line = lines[person_index - 1]
        person_parts = person_line.strip().split()
        px, py, pw, ph = map(float, person_parts[1:])

        # initialize PPE counts and labels
        ppe_counts = {ppe: 0 for ppe in ppe_max_count.keys()}
        ppe_labels = {ppe: [] for ppe in ppe_max_count.keys()}

        for line in lines:
            parts = line.strip().split()
            if parts[0] != "0":  # skip 'person' class
                class_id = int(parts[0])
                ppe_type = classes[class_id]
                
                # convert coordinates relative to the cropped person
                x, y, w, h = map(float, parts[1:])
                new_x = (x - px + pw/2) / pw
                new_y = (y - py + ph/2) / ph
                new_w = w / pw
                new_h = h / ph

                # ensure the coordinates are within the cropped person boundaries
                new_x = max(0, min(1, new_x))
                new_y = max(0, min(1, new_y))
                new_w = max(0, min(1, new_w))
                new_h = max(0, min(1, new_h))

                # store the label with the new coordinates
                ppe_labels[ppe_type].append((new_x, new_y, new_w, new_h, f"{class_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n"))

        # filter the labels based on the maximum count for each PPE type
        for ppe_type, labels in ppe_labels.items():
            if len(labels) > ppe_max_count[ppe_type]:
                def center_distance(label):
                    x, y, w, h, _ = label
                    x_center = x + w / 2
                    y_center = y + h / 2
                    return (x_center - 0.5)**2 + (y_center - 0.5)**2  # distance to the center

                # sort the labels based on the distance to the center
                sorted_labels = sorted(labels, key=center_distance)
                
                # take the closest labels to the center up to the max count
                valid_labels = sorted_labels[:ppe_max_count[ppe_type]]
            else:
                # keep all labels if the count is less than or equal to the max count
                valid_labels = labels

            # write the valid labels to the output file
            for label in valid_labels:
                outfile.write(label[4])

def create_data_yml(output_folder, classes):
    def get_relative_path(base_path, target_path):
        return os.path.relpath(target_path, base_path)

    output_folder_abs = os.path.abspath(output_folder)
    train_images_path = os.path.join(output_folder_abs, 'train', 'images')
    val_images_path = os.path.join(output_folder_abs, 'val', 'images')
    test_images_path = os.path.join(output_folder_abs, 'test', 'images')

    data_yml_content = f"""
names: {classes}
nc: {len(classes)}
test: {get_relative_path(output_folder_abs, test_images_path)}
train: {get_relative_path(output_folder_abs, train_images_path)}
val: {get_relative_path(output_folder_abs, val_images_path)}
    """

    with open(os.path.join(output_folder, 'data.yml'), 'w') as f:
        f.write(data_yml_content.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset for PPE detection")
    parser.add_argument("images_folder", type=str, help="Path to the folder containing images", default=r'datasets\voc_original\images', nargs='?')
    parser.add_argument("labels_folder", type=str, help="Path to the folder containing labels", default=r'datasets\yolo_annotations', nargs='?')
    parser.add_argument("output_folder", type=str, help="Path to the folder where the split dataset will be organized", default=r'datasets\yolo_ppe', nargs='?')
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion of the data to use for training")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proportion of the data to use for validation")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Proportion of the data to use for testing")

    args = parser.parse_args()

    # check if output folder exists
    if os.path.exists(args.output_folder):
        print(f"Output folder {args.output_folder} already exists. Please provide a new folder path.")
        exit(1)

    split_ratio = {'train': args.train_ratio, 'val': args.val_ratio, 'test': args.test_ratio}

    # step 1: crop images and filter annotations
    temp_images_folder, temp_labels_folder = crop_person_images_and_filter_annotations(
        args.images_folder, args.labels_folder, args.output_folder
    )

    # step 2: split the dataset
    split_dataset(temp_images_folder, temp_labels_folder, args.output_folder, split_ratio)

    # step 3: create data.yml
    create_data_yml(args.output_folder, classes)

    print(f"Dataset preparation completed. Output saved to {args.output_folder}")
