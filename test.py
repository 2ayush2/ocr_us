# test_ocr.py
import os
import json
import pandas as pd
from tamp_detect import (
    preprocess,
    ocr_pan,
    extract_data,
)  # Make sure tamp_detect has these functions

# Define paths
image_folder = "images"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# List all images in the sample_data folder
image_files = [
    f for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg", ".png"))
]

# Iterate through each image and extract data
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    print(f"\nProcessing image: {image_file}")

    # Preprocess the image
    gray_image, original_image = preprocess(image_path)

    # Perform OCR
    df = ocr_pan(gray_image)

    # Extract data
    extracted_data = extract_data(df)

    # Print extracted data
    print("\nExtracted Data:")
    for key, value in extracted_data.items():
        print(f"{key}: {value}")

    # Save extracted data to JSON
    json_file_path = os.path.join(
        output_folder, f"{os.path.splitext(image_file)[0]}_data.json"
    )
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)
    print(f"Extracted data saved to {json_file_path}")
