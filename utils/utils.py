import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(images_folder, labels_folder, output_folder, split_ratio={'train': 0.7, 'val': 0.2, 'test': 0.1}):
    """
    Splits images and labels into training, validation, and test sets.

    Parameters:
    - images_folder (str): Path to the folder containing images.
    - labels_folder (str): Path to the folder containing labels.
    - output_folder (str): Path to the folder where the split dataset will be organized.
    - split_ratio (dict): Dictionary with 'train', 'val', and 'test' ratios for splitting the dataset.
    """
    
    # read images and corresponding labels
    images = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    image_paths = [os.path.join(images_folder, img) for img in images]
    label_paths = [os.path.join(labels_folder, os.path.splitext(img)[0] + '.txt') for img in images]

    # print the total number of images
    total_images = len(image_paths)
    print(f"Total images: {total_images}")

    # split the dataset
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        image_paths, label_paths, test_size=1 - split_ratio['train'], random_state=42
    )

    # split ratio calculation for validation and test sets
    temp_val_ratio = split_ratio['val'] / (split_ratio['val'] + split_ratio['test'])

    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=(1 - temp_val_ratio), random_state=42
    )

    # print the sizes of the splits
    print(f"Train set size: {len(train_imgs)}")
    print(f"Validation set size: {len(val_imgs)}")
    print(f"Test set size: {len(test_imgs)}")

    # create the output folder structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_folder, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, split, 'labels'), exist_ok=True)

    # function to copy files
    def copy_files(image_list, label_list, split):
        for img_path, label_path in zip(image_list, label_list):
            shutil.copy(img_path, os.path.join(output_folder, split, 'images'))
            shutil.copy(label_path, os.path.join(output_folder, split, 'labels'))

    # Copy the files to the split folders
    copy_files(train_imgs, train_labels, 'train')
    copy_files(val_imgs, val_labels, 'val')
    copy_files(test_imgs, test_labels, 'test')

    # remove temporary folders after saving the split dataset
    shutil.rmtree(images_folder)
    shutil.rmtree(labels_folder)
    
    print(f"Dataset split and saved successfully at {output_folder}")