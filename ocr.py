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
image_path = "sample data/test.jpg"


# Enhanced preprocessing function
def preprocess(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
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
        M = cv2.getRotationMatrix2D(
            (gray.shape[1] // 2, gray.shape[0] // 2), angle, 1.0
        )
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
results = reader.readtext(gray_image, detail=1)
df = pd.DataFrame(results, columns=["bbox", "text", "conf"])

# Log raw OCR text for debugging
logging.info("Extracted Text from OCR:")
for _, row in df.iterrows():
    logging.info(row["text"])

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
image_path = "sample data/test.jpg"


# Enhanced preprocessing function
def preprocess(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
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
        M = cv2.getRotationMatrix2D(
            (gray.shape[1] // 2, gray.shape[0] // 2), angle, 1.0
        )
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
results = reader.readtext(gray_image, detail=1)
df = pd.DataFrame(results, columns=["bbox", "text", "conf"])

# Log raw OCR text for debugging
logging.info("Extracted Text from OCR:")
for _, row in df.iterrows():
    logging.info(row["text"])

# Extract data based on new PAN card format
extracted_data = {
    "PAN": "",
    "Name": "",
    "Address": "",
    "Date Of Birth": "",
    "ID No": "",
    "Issued Date": "",
    "Organization": "",
}

temp_address = []  # To store multi-line addresses

# Iterate through text to extract key-value pairs
for _, row in df.iterrows():
    text = row["text"].strip()

    # Log each text segment
    logging.info(f"Processing text: {text}")

    # Extract PAN Number
    if re.search(r"\bPAN[:\s]*\d{9}\b", text, re.IGNORECASE):
        extracted_data["PAN"] = re.search(r"\d{9}", text).group()

    # Extract Name (Improved Regex and Logic)
    elif re.search(r"(Name|नाम)[:\s]*", text, re.IGNORECASE):
        name_match = re.search(r"[:\s]*(.*)", text)
        if name_match:
            extracted_name = name_match.group(1).strip()
            if extracted_name and not extracted_data["Name"]:
                extracted_data["Name"] = extracted_name
            elif extracted_name:
                extracted_data["Name"] += " " + extracted_name

    # Extract multi-line Address
    elif re.search(r"\bAddress[:\s]*", text, re.IGNORECASE) or (
        extracted_data["Address"] and not re.search(r"\d{4}.\d{2}.\d{2}", text)
    ):
        temp_address.append(text)

    # Extract Date of Birth (Improved)
    elif re.search(
        r"(Date Of Birth|DOB|जन्म मिति)[:\s]*\d{4}[-./]\d{2}[-./]\d{2}",
        text,
        re.IGNORECASE,
    ):
        dob_match = re.search(r"\d{4}[-./]\d{2}[-./]\d{2}", text)
        if dob_match:
            extracted_data["Date Of Birth"] = dob_match.group().strip()

    # Extract ID No
    elif re.search(r"\bID No[:\s]*", text, re.IGNORECASE):
        extracted_data["ID No"] = text.split(":")[-1].strip()

    # Extract Issued Date (Enhanced)
    elif re.search(
        r"(Issued Date|जारी मिति)[:\s]*\d{4}[-./]\d{2}[-./]\d{2}", text, re.IGNORECASE
    ):
        issued_date_match = re.search(r"\d{4}[-./]\d{2}[-./]\d{2}", text)
        if issued_date_match:
            extracted_data["Issued Date"] = issued_date_match.group().strip()

    # Extract Organization
    elif re.search(r"करदाता सेवा कार्यालय", text, re.IGNORECASE):
        extracted_data["Organization"] = text.strip()

# Combine multi-line addresses
if temp_address:
    extracted_data["Address"] = " ".join(temp_address)

# Print Extracted Data in JSON Format
print("\nExtracted Data in JSON Format:")
print(json.dumps(extracted_data, indent=4, ensure_ascii=False))

# Save Extracted Data to a JSON File
json_file_path = os.path.join(output_folder, "extracted_pan_data.json")
with open(json_file_path, "w", encoding="utf-8") as json_file:
    json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)

print(f"\nExtracted data saved to {json_file_path}")

logging.info("OCR Extraction Completed Successfully")

# Print Extracted Data in JSON Format
print("\nExtracted Data in JSON Format:")
print(json.dumps(extracted_data, indent=4, ensure_ascii=False))

# Save Extracted Data to a JSON File
json_file_path = os.path.join(output_folder, "extracted_pan_data.json")
with open(json_file_path, "w", encoding="utf-8") as json_file:
    json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)

print(f"\nExtracted data saved to {json_file_path}")

logging.info("OCR Extraction Completed Successfully")
