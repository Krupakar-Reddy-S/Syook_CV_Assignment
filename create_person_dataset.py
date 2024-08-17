import os
import shutil
import argparse

from utils.utils import split_dataset

def filter_person_class(label_path, output_path):
    with open(label_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            if line.startswith("0 "): # search for 'person' class
                outfile.write(line)

def create_data_yml(output_folder):
    # only 'person' class is considered
    classes = ["person"]
    
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
    parser = argparse.ArgumentParser(description="Split and organize a YOLO dataset")
    parser.add_argument("images_folder", type=str, help="Path to the folder containing images", default=r'datasets\voc_original\images', nargs='?')
    parser.add_argument("labels_folder", type=str, help="Path to the folder containing labels", default=r'datasets\yolo_annotations', nargs='?')
    parser.add_argument("output_folder", type=str, help="Path to the folder where the split dataset will be organized", default=r'datasets\yolo_person', nargs='?')
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion of the data to use for training")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proportion of the data to use for validation")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Proportion of the data to use for testing")

    args = parser.parse_args()

    # check if output folder exists
    if os.path.exists(args.output_folder):
        print(f"Output folder {args.output_folder} already exists. Please provide a new folder path.")
        exit(1)

    split_ratio = {'train': args.train_ratio, 'val': args.val_ratio, 'test': args.test_ratio}

    # create temporary folders
    temp_images_folder = os.path.join(args.output_folder, 'temp_images')
    temp_labels_folder = os.path.join(args.output_folder, 'temp_labels')
    os.makedirs(temp_images_folder, exist_ok=True)
    os.makedirs(temp_labels_folder, exist_ok=True)

    # Step 1: copy images and filter labels
    for filename in os.listdir(args.images_folder):
        if filename.endswith('.jpg'):
            shutil.copy(os.path.join(args.images_folder, filename), os.path.join(temp_images_folder, filename))
    
    for filename in os.listdir(args.labels_folder):
        if filename.endswith('.txt'):
            label_path = os.path.join(args.labels_folder, filename)
            output_path = os.path.join(temp_labels_folder, filename)
            filter_person_class(label_path, output_path)

    # Step 2: split the dataset
    split_dataset(temp_images_folder, temp_labels_folder, args.output_folder, split_ratio)

    # Step 3: create data.yml
    create_data_yml(args.output_folder)
    
    print(f"Dataset preparation completed. Output saved to {args.output_folder}")
