import os

import cv2
import numpy as np
import pytesseract
from pathlib import Path
import json
import re
from pdf2image import convert_from_path
import sys
from pathlib import Path
from PIL import Image, ImageChops
import easyocr
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INPUT_FOLDER = r"C:\Users\gruia\Desktop\Disertatie\pdf - poze"
OUTPUT_FOLDER = r"C:\Users\gruia\Desktop\Disertatie\rezultate"

import os
from PIL import Image, ImageChops, ImageStat


def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_data = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')

    reader = easyocr.Reader(['en'])

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_dir, filename)

            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB")
                    width, height = img.size
                    left = int(width * 0.08)
                    top = int(height * 0.05)
                    right = int(width * 0.92)
                    bottom = int(height * 0.90)
                    img_main = img.crop((left, top, right, bottom))
                    w_main, h_main = img_main.size
                    split_line = int(h_main * 0.18)
                    title_section = img_main.crop((0, int(split_line * 0.20), w_main, int(split_line * 0.85)))
                    title_img_name = save_part(title_section, filename, "titlu", output_dir)

                    title_np = np.array(title_section)
                    title_np = cv2.cvtColor(title_np, cv2.COLOR_RGB2BGR)

                    title_np = cv2.resize(title_np, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    title_np = cv2.cvtColor(title_np, cv2.COLOR_BGR2GRAY)

                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    title_np = cv2.filter2D(title_np, -1, kernel)

                    results = reader.readtext(title_np, paragraph=True, contrast_ths=0.1, mag_ratio=2.0)

                    texts = [res[1] for res in results]
                    title_text = " ".join(texts)

                    clean_title = re.sub(r'^\d+[\s\.]+', '', title_text).strip()

                    page_info = {
                        "original_file": filename,
                        "title": clean_title,
                        "title_image": title_img_name,
                        "steps": []
                    }

                    steps_area = img_main.crop((0, split_line, w_main, h_main))
                    sw, sh = steps_area.size

                    for row in range(2):
                        for col in range(3):
                            c_left = col * (sw // 3)
                            c_top = row * (sh // 2)
                            c_right = (col + 1) * (sw // 3)
                            c_bottom = (row + 1) * (sh // 2)

                            step_cell = steps_area.crop((c_left, c_top, c_right, c_bottom))
                            step_num = row * 3 + col + 1

                            split_y = find_horizontal_split(step_cell)

                            drawing_part = step_cell.crop((0, 0, step_cell.width, split_y))
                            text_part = step_cell.crop((0, split_y, step_cell.width, step_cell.height))

                            draw_img_name = save_part(drawing_part, filename, f"pas_{step_num}_desen", output_dir)
                            text_img_name = save_part(text_part, filename, f"pas_{step_num}_text", output_dir)

                            text_np = cv2.cvtColor(np.array(text_part), cv2.COLOR_RGB2GRAY)
                            results_step = reader.readtext(text_np, paragraph=True)
                            step_description = " ".join([res[1] for res in results_step])

                            step_description = re.sub(r'^\d+[\s\.]+', '', step_description).strip()

                            page_info["steps"].append({
                                "step_number": step_num,
                                "description": step_description,
                                "drawing_image": draw_img_name,
                                "text_image": text_img_name
                            })

                    print(f"Procesat complet: {filename}")
                    all_data.append(page_info)
                    print(f"Procesat: {clean_title}")

            except Exception as e:
                print(f"Eroare la {filename}: {e}")
    json_output_path = os.path.join(output_dir, 'date_imagini.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

    print(f"\nProcesare finalizată! JSON salvat în: {json_output_path}")


def find_horizontal_split(cell):
    w, h = cell.size
    start_scan = int(h * 0.45)
    end_scan = int(h * 0.85)

    best_y = int(h * 0.70)
    max_whiteness = -1

    for y in range(start_scan, end_scan):
        line = cell.crop((0, y, w, y + 1))
        stat = ImageStat.Stat(line)
        avg_color = sum(stat.mean) / 3

        if avg_color > max_whiteness:
            max_whiteness = avg_color
            best_y = y
            if avg_color > 254:
                break
    return best_y
def save_part(part_img, original_name, suffix, output_dir):
    bg = Image.new(part_img.mode, part_img.size, (255, 255, 255))
    diff = ImageChops.difference(part_img, bg)
    bbox = diff.getbbox()

    if bbox:
        cropped = part_img.crop(bbox)
        p = 30
        result = Image.new("RGB", (cropped.width + p * 2, cropped.height + p * 2), "white")
        result.paste(cropped, (p, p))

        filename = f"{os.path.splitext(original_name)[0]}_{suffix}.png"
        result.save(os.path.join(output_dir, filename), "PNG")
        return filename
    return None

if __name__ == "__main__":
    folder_sursa = 'pdf - poze'
    folder_destinatie = 'rezultate'

    if os.path.exists(folder_sursa):
        process_images(folder_sursa, folder_destinatie)
    else:
        print(f"Eroare: Folderul '{folder_sursa}' nu a fost găsit!")