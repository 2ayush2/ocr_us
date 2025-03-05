import streamlit as st
import os
import cv2
import pandas as pd
from tamp_detect import (
    preprocess,
    ocr_pan,
    plot_bounding_boxes,
    extract_data,
)  # Corrected import


# Save extracted data to JSON
def save_to_json(extracted_data):
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    json_file_path = os.path.join(output_folder, "extracted_pan_data.json")
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        import json

        json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)
    st.success(f"Extracted data saved to {json_file_path}")


def main():
    st.title("PAN Card Text Detection using OCR")

    # Check if the 'uploads' folder exists, if not, create it
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    uploaded_file = st.sidebar.file_uploader(
        "Upload PAN Card Image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        global image_path
        image_path = os.path.join("uploads", uploaded_file.name)

        # Preprocess the image
        preprocessed_image, original_image = preprocess(image_path)

        # Extract text using OCR
        df = ocr_pan(preprocessed_image)

        # Display bounding boxes
        st.subheader("Bounding Boxes Plot")
        plot_bounding_boxes(df, image_path)

        # Extract and display structured data
        extracted_data = extract_data(df)

        # Display extracted data in a cleaner format
        st.subheader("Extracted Data:")
        for key, value in extracted_data.items():
            st.write(f"**{key}**: {value}")

        # Save extracted data to JSON
        save_to_json(extracted_data)

        # Remove uploaded image
        os.remove(image_path)
        st.sidebar.success("Uploaded image deleted successfully.")


if __name__ == "__main__":
    main()
