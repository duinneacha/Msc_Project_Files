import os
import random
import string
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from pathlib import Path


def random_color():
    return tuple(np.random.randint(0, 256, size=3))


def high_contrast_color(bg_color):
    return tuple(255 - c for c in bg_color)


def random_padding():
    return np.random.randint(5, 51, size=2)


def generate_digit_image(font_file, digit, output_folder):
    bg_color = random_color()
    text_color = high_contrast_color(bg_color)

    font_size = np.random.randint(20, 101)
    font = ImageFont.truetype(font_file, font_size)

    img_size = tuple(font.getsize(digit))
    img = Image.new("RGB", img_size, bg_color)

    draw = ImageDraw.Draw(img)
    draw.text((0, 0), digit, fill=text_color, font=font)

    angle = np.random.uniform(-30, 30)
    img = img.rotate(angle, expand=1, fillcolor=bg_color)

    img = img.crop((0, 0, *img_size))
    pad = random_padding()
    pad_left, pad_top = pad
    pad_right, pad_bottom = pad
    img = ImageOps.expand(
        img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=bg_color
    )

    # img = ImageOps.expand(img, border=pad, fill=bg_color)

    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    file_name = (
        "".join(random.choices(string.ascii_letters + string.digits, k=8))
        + f"_{digit}.png"
    )
    file_path = os.path.join(output_folder, file_name)
    img.save(file_path)

    return file_name


def create_digit_images(font_file, num_files_to_create, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    records = []

    for _ in range(num_files_to_create):
        digit = str(np.random.randint(0, 10))
        file_name = generate_digit_image(font_file, digit, output_folder)
        records.append({"file_name": file_name, "digit": digit})

    return pd.DataFrame(records)


def save_records_to_csv(records, csv_file):
    if os.path.exists(csv_file):
        existing_records = pd.read_csv(csv_file)
        records = pd.concat([existing_records, records], ignore_index=True)

    records.to_csv(csv_file, index=False)


def main(font_file, num_files_to_create, output_folder, csv_file):
    records = create_digit_images(font_file, num_files_to_create, output_folder)
    save_records_to_csv(records, csv_file)
