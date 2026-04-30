import os
import pytesseract
import cv2

input_folder = "bills_cleaned"
output_file = "bills_cleaned.txt"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_processor(input_folder, output_file):
    extracted_text = ""
    validate_Extensions = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(validate_Extensions):

            input_path = os.path.join(input_folder, filename)

            try:
                # Read processed image (not raw path laziness)
                img = cv2.imread(input_path)

                if img is None:
                    print(f"Failed to load {filename}")
                    continue

                config = "--oem 3 --psm 4"
                text = pytesseract.image_to_string(img, config=config)

                print("*" * 20)
                print(text.strip())
                print("*" * 20)

                extracted_text += f"Extracted text from {filename}:\n{text.strip()}\n\n"

            except Exception as e:
                print(f"Problem processing {filename}: {e}")

    # WRITE ONCE (like a civilized program)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_text)

        
if __name__ == "__main__":
    ocr_processor(input_folder, output_file)

