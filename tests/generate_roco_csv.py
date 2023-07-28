import csv
import pickle
import json
import os
import torchvision
from tqdm import tqdm
import argparse


def main(dataset_dir: str, input_text_name: str, out_dir: str, out_csv_file: str):
    
    """
    The datest directory should have the following structure:
        dataset_dir/
            ├── input_text_name.txt (E.g. captions.txt)
            └── images/
    In each "images" folder, filenames of images are: "ROCO_00020.jpg", "ROCO_00027.jpg", etc...
    
    In input_text_name.txt (E.g. captions.txt), the content is stored as below:
    ROCO_00020	 Axial computed tomography scan of the pelvis showing a diffuse infiltration of the bladder wall, catheter in situ (arrow).
    ROCO_00027	 Postoperative anteroposterior radiograph of the pelvis.
    """
    image_dir = os.path.join(dataset_dir, "images")
    text_file_path = os.path.join(dataset_dir, input_text_name) # captions.txt
    out_csv_path = os.path.join(out_dir, out_csv_file)

    # A dictionary to store the filepaths and captions
    filepaths = {}
    captions = {}

    # Read the captions
    with open(text_file_path, "r") as captions_file:
        lines = captions_file.readlines()

    for line in tqdm(lines, desc="Processing captions", unit="lines"):
        # Split the line into the image ID and caption
        try:
            image_id, caption = line.strip().split('\t')
            # if image_id == 'ROCO_00059':
            #     print(image_id, caption)
        except:
            continue

        # Processing the caption content
        caption = caption.lower().rstrip().replace("\\n", "").rstrip(".")
        # if image_id == 'ROCO_00059':
        #     print(image_id, caption)
        try:
            caption = caption.encode('ascii', 'ignore').decode('ascii')
        except:
            continue
        #Skip if the caption is too short
        # if len(caption) < 10:
        #     continue
        # if len(caption.split()) < 5:
        #     continue

        # Construct the path to the image file
        image_path = os.path.join(image_dir, f'{image_id}.jpg')

        # Check if the image file exists
        if not os.path.exists(image_path):
            continue
        
        # to make sure the file is a valid image
        try:
            temp_data = torchvision.io.image.read_file(image_path)
        except:
            print(image_path)
            continue

        # Add the image path and caption to the captions dictionary
        captions[image_id] = caption
        filepaths[image_id] = image_path
        # if image_id == 'ROCO_00059':
        #     print(filepaths)
        #     print(captions)

    # Prepare data for CSV
    data = [{"filepath": filepaths[key], "caption": captions[key]} for key in filepaths.keys()]


    # Write the CSV file
    with open(out_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filepath', 'caption']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    """
    Example: 
    python generate_roco_csv.py --dataset_dir "/mnt/eds_ml/Users/Yilu_ML/roco-dataset/data/train/radiology" --input_text_name "captions.txt" --out_dir "./data" --out_csv_file "roco_train.csv"
    python generate_roco_csv.py --dataset_dir "/mnt/eds_ml/Users/Yilu_ML/roco-dataset/data/validation/radiology" --input_text_name "captions.txt" --out_dir "./data" --out_csv_file "roco_validation.csv"
    python generate_roco_csv.py --dataset_dir "/mnt/eds_ml/Users/Yilu_ML/roco-dataset/data/test/radiology" --input_text_name "captions.txt" --out_dir "./data" --out_csv_file "roco_test.csv"
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default=r"E:\Work\roco-dataset\data\validation\radiology")
    parser.add_argument('--input_text_name', default="captions.txt") 
    parser.add_argument('--out_dir', default=".\\data")
    parser.add_argument('--out_csv_file', default="roco_validation.csv")

    args = parser.parse_args()
    exit(main(args.dataset_dir, args.input_text_name, args.out_dir, args.out_csv_file))