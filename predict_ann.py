import json
import os
from PIL import Image
from argparse_utils import parse_args


def predict_ann(img_dir):
    # Define the categories for the COCO dataset
    categories = [{"id": 0, "name": "cluster"}]

    # Define the COCO dataset dictionary
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    # Loop through the images in the input directory
    for idx, image_file in enumerate(os.listdir(img_dir)):
        
        # Load the image and get its dimensions
        image_path = os.path.join(img_dir, image_file)
        image = Image.open(image_path)
        width, height = image.size
        
        # Add the image to the COCO dataset
        image_dict = {
            "id": idx,
            "width": width,
            "height": height,
            "file_name": image_file
        }
        coco_dataset["images"].append(image_dict)

    # Save the COCO dataset to a JSON file
    with open(os.path.join(img_dir, '_annotations.coco.json'), 'w') as f:
        json.dump(coco_dataset, f)


if __name__=='__main__':
    args = parse_args()
    predict_ann(args.infer_directory)
