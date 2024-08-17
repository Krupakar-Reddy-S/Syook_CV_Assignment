import os
import sys

# add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from utils.tests import *

# path of directory containing this script
this_dir = os.path.dirname(os.path.abspath(__file__))

# path to the train directory of the dataset
parent_dir = os.path.join(this_dir, r"../../datasets/yolo_person/train")

# path to the data.yml file
data_yml_path = os.path.join(this_dir, r"../../datasets/yolo_person/data.yml")

class_names = load_class_names(data_yml_path)
images_dir = os.path.join(parent_dir, 'images')
labels_dir = os.path.join(parent_dir, 'labels')
output_dir = os.path.join(this_dir, 'output')

process_images(images_dir, labels_dir, output_dir, class_names, class_colors)