# OpenMLkit OCR

A lightweight, offline Python OCR (Optical Character Recognition) library utilizing highly optimized, mobile-ready Google ML Kit TFLite models. It performs text detection using a Region Proposal Network (RPN) and line recognition using a CRNN-CTC architecture.

---

## Features
- **Fully Offline:** Runs entirely local, no API keys or internet connection required after downloading models.
- **Multilingual Support:** Supports 15+ languages and scripts (Cyrillic/Russian, Latin/English, Chinese, Japanese, Korean, Arabic, Hebrew, and various Indian scripts).
- **Auto-downloading:** Automatically downloads and caches required models from Hugging Face if they are not present locally.
- **High Quality Stitching:** Handles wide text lines without squishing by dividing them into overlapping chunks and merging them using fuzzy suffix-prefix alignment.

---

## Installation

Install the package directly using pip (or clone the repository):

```bash
pip install "numpy<2.0.0" opencv-python tflite-runtime huggingface_hub
```

To install OpenMLkit as a package from source:
```bash
git clone https://github.com/0cve0/OpenMLkitOCR.git
cd OpenMLkitOCR
pip install -e .
```

---

## Quick Start

```python
import os
import cv2
from openmlkit import OpenMLKitOCR

# Configure Hugging Face model source (or use defaults)
os.environ["OPENMLKIT_MODEL_REPO"] = "0cve0/OpenMLKitOCR"

# Initialize OCR pipeline for Cyrillic (Russian) text
ocr = OpenMLKitOCR(lang='ru')

# Load image
img = cv2.imread("scratch/russian_test.png")

# Run OCR (detect and recognize text)
results = ocr.run(img, score_threshold=0.35)

# Output results
for r in results:
    print(f"Box: {r['box']} -> Text: {r['text']}")
```

---

## Project Structure

- `openmlkit/` - Core Python package directory.
  - `detector.py` - RPN text detection logic.
  - `recognizer.py` - CRNN text recognition logic.
  - `labelmap.py` - Parse binary protobuf label maps.
  - `pipeline.py` - OCR pipeline joining detection, tiling, and recognition.

---

## License

This project is licensed under the Apache 2.0 License. The model weights are subject to Google's terms and licenses for ML Kit.
