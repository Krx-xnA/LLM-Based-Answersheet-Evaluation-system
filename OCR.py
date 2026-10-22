#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
OCR Module - LLM Based Answer Sheet Evaluation System

Pipeline:
PDF/Image
    ↓
Image Preprocessing
    ↓
Text Line Detection
    ↓
TrOCR
    ↓
Extracted Text
"""

import os
import cv2
import torch
import numpy as np

from PIL import Image
from pdf2image import convert_from_path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/trocr-base-handwritten"

# Automatically use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD TROCR MODEL
# ============================================================

print(f"Loading TrOCR model on {DEVICE}...")

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()

print("TrOCR model loaded successfully.")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Preprocess an answer-sheet image before line segmentation.

    Parameters
    ----------
    image : PIL.Image
        Input answer-sheet image.

    Returns
    -------
    numpy.ndarray
        Thresholded image.
    """

    # Convert PIL image to NumPy array
    image = np.array(image.convert("RGB"))

    # Convert RGB -> grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Light noise reduction
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive thresholding helps with uneven lighting
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    return binary


# ============================================================
# TEXT LINE DETECTION
# ============================================================

def detect_text_lines(original_image, binary_image):
    """
    Detect approximate handwritten text lines from an answer sheet.

    Parameters
    ----------
    original_image : PIL.Image
        Original answer-sheet image.

    binary_image : numpy.ndarray
        Preprocessed binary image.

    Returns
    -------
    list
        List of cropped PIL images containing individual text lines.
    """

    # Horizontal kernel connects words belonging to the same line
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (40, 5)
    )

    dilated = cv2.dilate(
        binary_image,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore tiny regions/noise
        if w > 30 and h > 10:
            boxes.append((x, y, w, h))

    # Sort from top to bottom
    boxes = sorted(boxes, key=lambda box: box[1])

    original_np = np.array(original_image.convert("RGB"))

    line_images = []

    for x, y, w, h in boxes:

        # Add padding around handwriting
        padding = 10

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(original_np.shape[1], x + w + padding)
        y2 = min(original_np.shape[0], y + h + padding)

        crop = original_np[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = Image.fromarray(crop)

        line_images.append(crop)

    return line_images


# ============================================================
# TROCR TEXT RECOGNITION
# ============================================================

def recognize_line(line_image):
    """
    Run TrOCR on one handwritten text-line image.

    Parameters
    ----------
    line_image : PIL.Image
        Image containing one handwritten text line.

    Returns
    -------
    str
        Recognized text.
    """

    line_image = line_image.convert("RGB")

    pixel_values = processor(
        images=line_image,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()


# ============================================================
# OCR FOR ONE IMAGE
# ============================================================

def extract_text_from_image(image):
    """
    Extract handwritten text from a PIL image.
    """

    binary_image = preprocess_image(image)

    text_lines = detect_text_lines(
        image,
        binary_image
    )

    extracted_lines = []

    print(f"Detected {len(text_lines)} text regions.")

    for index, line_image in enumerate(text_lines, start=1):

        try:

            text = recognize_line(line_image)

            if text:

                extracted_lines.append(text)

                print(
                    f"Line {index}/{len(text_lines)}: {text}"
                )

        except Exception as error:

            print(
                f"Could not process line {index}: {error}"
            )

    return "\n".join(extracted_lines)


# ============================================================
# PDF OCR
# ============================================================

def extract_text_from_pdf(pdf_path, dpi=300):
    """
    Convert PDF pages to images and perform OCR.

    Parameters
    ----------
    pdf_path : str
        Path to PDF answer sheet.

    dpi : int
        Resolution used while converting PDF pages.

    Returns
    -------
    str
        Extracted text from all pages.
    """

    print("Converting PDF into images...")

    pages = convert_from_path(
        pdf_path,
        dpi=dpi
    )

    complete_text = []

    for page_number, page in enumerate(pages, start=1):

        print(
            f"\nProcessing page {page_number}/{len(pages)}..."
        )

        page_text = extract_text_from_image(page)

        complete_text.append(
            f"\n--- Page {page_number} ---\n{page_text}"
        )

    return "\n".join(complete_text)


# ============================================================
# MAIN OCR FUNCTION
# ============================================================

def extract_text(file_path):
    """
    Main OCR function.

    Supports:
    - JPG
    - JPEG
    - PNG
    - PDF

    Example
    -------
    text = extract_text("answer_sheet.pdf")
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = os.path.splitext(file_path)[1].lower()

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if extension in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:

        print(f"Processing image: {file_path}")

        image = Image.open(file_path)

        return extract_text_from_image(image)

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    elif extension == ".pdf":

        print(f"Processing PDF: {file_path}")

        return extract_text_from_pdf(file_path)

    else:

        raise ValueError(
            "Unsupported file format. "
            "Please provide JPG, JPEG, PNG, TIFF, BMP, or PDF."
        )


# In[ ]:


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    # Change this to your answer-sheet path
    FILE_PATH = "sample_answer.jpg"

    try:

        result = extract_text(FILE_PATH)

        print("\n" + "=" * 60)
        print("EXTRACTED ANSWER")
        print("=" * 60)

        print(result)

        print("=" * 60)

    except Exception as error:

        print(f"\nOCR Error: {error}")

