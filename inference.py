import os
import cv2
import shutil
import argparse

from ultralytics import YOLO
from utils.vars import class_colors

# PPE classes in the dataset
ppe_classes = [
    'boots',
    'gloves',
    'hard-hat',
    'mask',
    'ppe-suit',
    'vest'
]

def perform_inference(input_dir, output_dir, person_det_model, ppe_detection_model):
    if os.path.isdir(input_dir) is False:
        raise ValueError(f"Input directory '{input_dir}' does not exist")
    
    # if the output directory exists, remove it
    if os.path.isdir(output_dir) is True:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # load the models
    person_model = YOLO(person_det_model)
    ppe_model = YOLO(ppe_detection_model)

    # get image files from the input directory
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    for image_file in image_files:
        image = cv2.imread(image_file)
        height, width, _ = image.shape

        # Step 1: person detection on the full image to get bounding boxes
        person_results = person_model(image)
        person_bboxes = person_results[0].boxes.xyxy.cpu().numpy()

        for i, bbox in enumerate(person_bboxes):
            xmin, ymin, xmax, ymax = map(int, bbox[:4])
            cropped_image = image[ymin:ymax, xmin:xmax]

            # Step 2: PPE detection on the cropped image
            ppe_results = ppe_model(cropped_image)
            ppe_bboxes = ppe_results[0].boxes.xyxy.cpu().numpy()
            ppe_confs = ppe_results[0].boxes.conf.cpu().numpy()
            ppe_classes_ids = ppe_results[0].boxes.cls.cpu().numpy()

            # Step 3: convert PPE bounding boxes to full image coordinates and draw them
            for j, ppe_bbox in enumerate(ppe_bboxes):
                ppe_xmin, ppe_ymin, ppe_xmax, ppe_ymax = map(int, ppe_bbox[:4])
                ppe_class_id = int(ppe_classes_ids[j]) if len(ppe_classes_ids) > j else -1
                ppe_confidence = float(ppe_confs[j]) if len(ppe_confs) > j else 0.0

                # convert to full image coordinates
                ppe_xmin_full = ppe_xmin + xmin
                ppe_ymin_full = ppe_ymin + ymin
                ppe_xmax_full = ppe_xmax + xmin
                ppe_ymax_full = ppe_ymax + ymin
                
                class_name = "Unknown"
                color = (0, 0, 0)

                # get class name and color for the bounding box
                if 0 <= ppe_class_id < len(ppe_classes):
                    class_name = ppe_classes[ppe_class_id]
                    color = class_colors.get(class_name, (0, 0, 0)) # defaults to black if class name is not found

                # draw the bounding boxes and labels on the full image
                cv2.rectangle(image, (ppe_xmin_full, ppe_ymin_full), (ppe_xmax_full, ppe_ymax_full), color, 2)
                label = f"{class_name}: {ppe_confidence:.2f}"
                cv2.putText(image, label, (ppe_xmin_full, ppe_ymin_full - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Step 4: save the final image with bounding boxes
        output_image_path = os.path.join(output_dir, os.path.basename(image_file))
        cv2.imwrite(output_image_path, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on images using YOLOv8 models")
    parser.add_argument("--input_dir", type=str, default=r'test\inference\input', help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default=r'test\inference\output', help="Directory to save output images with bounding boxes")
    parser.add_argument("--person_det_model", type=str, default=r'weights\person_200_original.pt', help="Path to the trained person detection model")
    parser.add_argument("--ppe_detection_model", type=str, default=r'weights\ppe_200_original.pt', help="Path to the trained PPE detection model")

    args = parser.parse_args()

    perform_inference(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)

    print(f"Inference completed. Results saved in {args.output_dir}")
