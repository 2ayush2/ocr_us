import pandas as pd
import numpy as np
import re
import cv2
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import easyocr
from datetime import datetime
import logging

# Configure logging
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(output_folder, "ocr_log.txt"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("OCR Extraction Started")

plt.style.use("ggplot")
reader = easyocr.Reader(["en", "hi"], gpu=False)

# Define image path
image_path = "sample data/pan1_sample.png"


# Enhanced preprocessing function
def preprocess(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Remove noise and improve clarity
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=30)

    # Fix skewed or rotated images using edge detection
    edges = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        M = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), angle, 1.0)
        gray = cv2.warpAffine(
            gray,
            M,
            (gray.shape[1], gray.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # Adaptive thresholding and dilation for better OCR
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)

    return gray, image


# Preprocess image
gray_image, original_image = preprocess(image_path)

# Extract text using EasyOCR
results = reader.readtext(gray_image, detail=1, paragraph=False)
df = pd.DataFrame(results, columns=["bbox", "text", "conf"])

# Log raw OCR text for debugging
logging.info("Extracted Text from OCR:")
for _, row in df.iterrows():
    logging.info(row["text"])

# Extract data based on new PAN card format
# Extract data based on new PAN card format
extracted_data = {
    "PAN": "Not Found",
    "Name": "Not Found",
    "Address": "Not Found",
    "Date Of Birth": "Not Found",
    "ID No": "Not Found",
    "Issued Date": "Not Found",
    "Organization": "Not Found",
}

temp_address = []  # To store multi-line addresses
temp_name = []  # To store multi-line names

# Enhanced regex patterns for accurate extraction with variations
pan_pattern = r"\b(?:PAN|P4N|PАН|P/N)[: \t]*([0-9A-Za-z]{9})\b"
name_pattern = (
    r"(?:Name|नाम)?[: \t]*([A-Za-z\s.]+)"  # Captures Name with or without colon
)
address_pattern = r"(?:Address|Addressः|Aadess|ठेगाना)[: \t]*([A-Za-z0-9\s\-\.,]*)"
dob_pattern = r"(?:Date Of Birth|DOB|जन्म मिति)[: \t]*([\d./-]+)"
id_no_pattern = r"(?:ID No|आईडी नं)[: \t]*([A-Za-z0-9-]+)"
issued_date_pattern = r"(?:Issued Date|Issued Date:|जारी मिति)?[: \t]*([\d./-]+)"
org_pattern = (
    r"(?:Organization|Organization;|संस्था|करदाता सेवा कार्यालय)[: \t]*([A-Za-z\s\-]*)"
)

# Print all extracted text for debugging
print("\nExtracted Text from OCR:")
for _, row in df.iterrows():
    print(f"Detected Text: {row['text']}")

# Iterate through text to extract key-value pairs
for _, row in df.iterrows():
    text = row["text"].strip()

    # Extract PAN
    if re.search(pan_pattern, text):
        extracted_data["PAN"] = re.search(pan_pattern, text).group(1)

    # Extract Name (even without colon)
    elif re.search(name_pattern, text, re.IGNORECASE):
        name = re.search(name_pattern, text).group(1).strip()
        temp_name.append(name)

    # Extract Address
    elif re.search(address_pattern, text, re.IGNORECASE):
        address = re.search(address_pattern, text).group(1).strip()
        if address:
            temp_address.append(address)

    # Extract Date of Birth
    elif re.search(dob_pattern, text, re.IGNORECASE):
        extracted_data["Date Of Birth"] = re.search(dob_pattern, text).group(1).strip()

    # Extract ID No
    elif re.search(id_no_pattern, text, re.IGNORECASE):
        extracted_data["ID No"] = re.search(id_no_pattern, text).group(1).strip()

    # Extract Issued Date (standalone date support)
    elif re.search(issued_date_pattern, text, re.IGNORECASE):
        date_match = re.search(issued_date_pattern, text)
        if date_match:
            extracted_data["Issued Date"] = date_match.group(1).strip()

    # Extract Organization (even with semicolon)
    elif re.search(org_pattern, text, re.IGNORECASE):
        extracted_data["Organization"] = re.search(org_pattern, text).group(1).strip()

# Combine multi-line names and addresses
if temp_name:
    extracted_data["Name"] = " ".join(temp_name)
if temp_address:
    extracted_data["Address"] = " ".join(temp_address)

# Ensure no field is blank, set to "Not Found" if empty
for key in extracted_data:
    if not extracted_data[key].strip():
        extracted_data[key] = "Not Found"

# Print final extracted data for debugging
print("\nFinal Extracted Data:")
for key, value in extracted_data.items():
    print(f"{key}: {value}")


# Save Extracted Data to a JSON File
json_file_path = os.path.join(output_folder, "extracted_pan_data.json")
with open(json_file_path, "w", encoding="utf-8") as json_file:
    json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)

print(f"\nExtracted data saved to {json_file_path}")

logging.info("OCR Extraction Completed Successfully")
