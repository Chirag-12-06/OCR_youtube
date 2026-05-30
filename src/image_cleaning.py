import os
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FOLDER = PROJECT_ROOT / "inputs" / "bills"
OUTPUT_FOLDER = PROJECT_ROOT / "inputs" / "bills_cleaned"


def image_cleaning(input_folder, output_folder):
    validate_Extensions = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(validate_Extensions):

            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                img = cv2.imread(input_path)

                if img is None:
                    print(f"Failed to load {filename}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Choose ONE denoising
                denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

                # Choose threshold dynamically
                if np.std(gray) < 50:
                    _, thresh = cv2.threshold(
                        denoised, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                else:
                    thresh = cv2.adaptiveThreshold(
                        denoised, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        11, 2
                    )

                # Optional dilation
                if np.mean(thresh) < 127:
                    thresh = cv2.dilate(thresh, np.ones((2,2), np.uint8), iterations=1)

                # Ensure correct polarity
                if np.mean(thresh) > 127:
                    thresh = cv2.bitwise_not(thresh)

                # Resize
                resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                cv2.imwrite(output_path, resized)
                print(f"Processed: {filename}")

            except Exception as e:
                print(f"Problem processing {filename}: {e}")


if __name__ == "__main__":
    image_cleaning(INPUT_FOLDER, OUTPUT_FOLDER)
