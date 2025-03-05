# importing libraries
import pandas as pd
import numpy as np
import re
import cv2
import streamlit as st
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import easyocr

plt.style.use("ggplot")

# Creating object of easyocr
reader = easyocr.Reader(
    ["en", "hi"], gpu=False
)  # Change gpu to False if you don't have a compatible GPU


# Preprocessing the orientation of the image
def preprocess(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Enhance contrast
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=20)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding for better results
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Denoise the image to reduce noise for OCR
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    return gray, image


# OCR function for PAN card
def ocr_pan(image):
    results = reader.readtext(image)
    df = pd.DataFrame(results, columns=["bbox", "text", "conf"])
    return df


# Extract data from OCR results
def extract_data(df):
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
    name_pattern = r"(?:Name|नाम)[: \t]*([A-Za-z\s.अ-हाि-्]*)"
    address_pattern = r"(?:Address|Addressः|Aadess|ठेगाना)[: \t]*(.*)"
    dob_pattern = r"(?:Date Of Birth|DOB|जन्म मिति)[: \t]*([\d./-]+)"
    id_no_pattern = r"(?:ID No|आईडी नं)[: \t]*([A-Za-z0-9-]+)"
    issued_date_pattern = r"(?:Issued Date|जारी मिति)?[: \t]*([\d./-]+)"
    org_pattern = r"(?:Organization|Organization;|संस्था|करदाता सेवा कार्यालय)[: \t]*(.*)"

    # Print all extracted text for debugging
    print("\nExtracted Text from OCR:")
    for _, row in df.iterrows():
        print(f"Detected Text: {row['text']}")

    # Iterate through text to extract key-value pairs
    for idx, row in df.iterrows():
        text = row["text"].strip()

        # Extract PAN
        if re.search(pan_pattern, text):
            extracted_data["PAN"] = re.search(pan_pattern, text).group(1)

        # Extract Name (even if it appears on the next line)
        elif re.search(r"(?:Name|नाम)[:\s]*$", text, re.IGNORECASE):
            next_index = idx + 1
            if next_index < len(df):
                next_text = df.iloc[next_index]["text"].strip()
                # Include Nepali characters in the name pattern
                if re.match(r"^[A-Za-z\s.अ-हाि-्]+$", next_text):
                    temp_name.append(next_text)
        elif re.search(name_pattern, text, re.IGNORECASE):
            name = re.search(name_pattern, text).group(1).strip()
            if name:
                temp_name.append(name)

        # Extract Address (handle variations like "ः")
        elif re.search(address_pattern, text, re.IGNORECASE):
            address = re.search(address_pattern, text).group(1).strip()
            if address and address != "ः":
                temp_address.append(address)

        # Extract Date of Birth
        elif re.search(dob_pattern, text, re.IGNORECASE):
            extracted_data["Date Of Birth"] = (
                re.search(dob_pattern, text).group(1).strip()
            )

        # Extract ID No
        elif re.search(id_no_pattern, text, re.IGNORECASE):
            extracted_data["ID No"] = re.search(id_no_pattern, text).group(1).strip()

        # Extract Issued Date (standalone date support)
        if re.search(r"^\d{4}[-./]\d{2}[-./]\d{2}$", text):
            extracted_data["Issued Date"] = text.strip()
        elif re.search(issued_date_pattern, text, re.IGNORECASE):
            date_match = re.search(issued_date_pattern, text)
            if date_match:
                extracted_data["Issued Date"] = date_match.group(1).strip()

        # Extract Organization
        elif re.search(org_pattern, text, re.IGNORECASE):
            organization = re.search(org_pattern, text).group(1).strip()
            if organization:
                extracted_data["Organization"] = organization

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

    return extracted_data


# Function to plot bounding boxes
def plot_bounding_boxes(df, image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Add bounding boxes to the image
    for index, row in df.iterrows():
        # Get bounding box coordinates
        bbox = row["bbox"]
        x1, y1 = map(int, bbox[0])
        x2, y2 = map(int, bbox[2])

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the image with bounding boxes using Streamlit
    st.image(
        image,
        channels="BGR",
        caption="Original Image with Bounding Boxes",
        use_column_width=True,
    )


# Show extracted data
def show_extracted_data(extracted_data):
    st.subheader("Extracted Data:")
    for key, value in extracted_data.items():
        st.write(f"{key}: {value}")


# Save extracted data to JSON
def save_to_json(extracted_data):
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    json_file_path = os.path.join(output_folder, "extracted_pan_data.json")
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)
    print(f"Extracted data saved to {json_file_path}")


# Main function
def main(image_path):
    # Preprocess the image
    gray_image, original_image = preprocess(image_path)

    # Perform OCR
    df = ocr_pan(gray_image)

    # Extract data
    extracted_data = extract_data(df)

    # Display extracted data
    show_extracted_data(extracted_data)

    # Save extracted data to JSON
    save_to_json(extracted_data)

    # Plot bounding boxes
    plot_bounding_boxes(df, image_path)


# Run the main function
if __name__ == "__main__":
    image_path = "images"
    main(image_path)
