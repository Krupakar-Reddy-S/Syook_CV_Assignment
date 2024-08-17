import os
import argparse
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(voc_folder, yolo_folder):
    # path to classes.txt in voc_folder
    classes_path = os.path.join(voc_folder, 'classes.txt')
    
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # create output directory if it does not exist
    if not os.path.exists(yolo_folder):
        os.makedirs(yolo_folder)
    else:
        print(f"Directory {yolo_folder} already exists. Exiting...")
        return

    # path to labels folder in voc_folder in PascalVOC format
    labels_folder = os.path.join(voc_folder, 'labels')
    
    for filename in os.listdir(labels_folder):
        if filename.endswith('.xml'):
            xml_path = os.path.join(labels_folder, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            image_width = int(root.find('size/width').text)
            image_height = int(root.find('size/height').text)

            # create corresponding YOLO annotation file
            yolo_filename = os.path.splitext(filename)[0] + '.txt'
            yolo_path = os.path.join(yolo_folder, yolo_filename)
            
            with open(yolo_path, 'w') as yolo_file:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in classes:
                        continue
                    class_id = classes.index(class_name)

                    # get bounding box coordinates
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)

                    # convert to YOLO format (normalized coordinates)
                    x_center = (xmin + xmax) / 2 / image_width
                    y_center = (ymin + ymax) / 2 / image_height
                    width = (xmax - xmin) / image_width
                    height = (ymax - ymin) / image_height

                    # write converted labels to YOLO annotation file
                    yolo_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC annotations to YOLOv8 format")
    parser.add_argument("voc_folder", type=str, help="Path to the folder containing PascalVOC XML annotations and classes.txt", default=r'datasets\voc_original', nargs='?')
    parser.add_argument("yolo_folder", type=str, help="Path to the folder to save YOLOv8 annotations", default=r'datasets\yolo_annotations', nargs='?')

    args = parser.parse_args()
    
    if not os.path.exists(args.voc_folder):
        print(f"Directory {args.voc_folder} does not exist. Exiting...")
        exit(1)
    
    convert_voc_to_yolo(args.voc_folder, args.yolo_folder)