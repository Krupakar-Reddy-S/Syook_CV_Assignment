import os
import argparse
from ultralytics import YOLO

def get_model_path(model_type, size):
    """Return the correct model path based on model type and size."""
    if model_type == 'person':
        if size == '50':
            model_filename = 'person_50_original.pt'
        elif size == '100':
            model_filename = 'person_100_resized.pt'
        elif size == '200':
            model_filename = 'person_200_original.pt'
    elif model_type == 'ppe':
        if size == '50':
            model_filename = 'ppe_50_original.pt'
        elif size == '100':
            model_filename = 'ppe_100_resized.pt'
        elif size == '200':
            model_filename = 'ppe_200_original.pt'
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', model_filename)
    return model_path

def get_data_path(model_type):
    """Return the correct dataset path based on the model type."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    
    if model_type == 'person':
        data_filename = r'datasets\Person_Detetction.v1i.yolov8\data.yaml'
    elif model_type == 'ppe':
        data_filename = r'datasets\PPE_Detection.v1i.yolov8\data.yaml'
    
    data_path = os.path.join(cur_dir, data_filename)
    return data_path

def main():
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument('model_type', choices=['person', 'ppe'], help="Specify the model type: 'person' or 'ppe'.")
    parser.add_argument('size', choices=['50', '100', '200'], help="Specify the model size: '50', '100', or '200'.")
    args = parser.parse_args()

    # get model path based on arguments
    model_path = get_model_path(args.model_type, args.size)
    
    # load the model
    model = YOLO(model_path)

    # get dataset path
    data_path = get_data_path(args.model_type)

    # run model validation
    results = model.val(data=data_path, save_json=True, verbose=True)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
