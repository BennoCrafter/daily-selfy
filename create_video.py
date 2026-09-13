import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import cv2
import ffmpeg
import numpy as np
from moviepy import AudioFileClip, VideoFileClip
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False
    print("[warning] pillow-heif not installed — .heic files will be skipped.")
    print("          Install with: pip install pillow-heif")


def get_image_date(img_name):
    path = os.path.join(image_folder, img_name)
    try:
        with Image.open(path) as img:
            return get_real_date(path, img)
    except Exception:
        # fallback to modification time
        return datetime.fromtimestamp(os.path.getmtime(path))


# ===== SETTINGS =====
image_folder = "resources/images"
music_file = "resources/music.mp3"
temp_video = "output/temp.mp4"
output_video = "output/output.mp4"
seconds_per_image = 0.25
fps = 30

WIDTH = 4032
HEIGHT = 3024

font_size = 120
margin = 40
outline_size = 4
# text_color = (255, 220, 0)
text_color = (137, 180, 250)

school_year_dict = {
    (date(2025, 9, 16), date(2026, 7, 31)): "10",
    (date(2026, 9, 15), date(2027, 7, 30)): "11",
    (date(2027, 9, 14), date(2028, 7, 28)): "12",
    (date(2028, 9, 12), date(2029, 7, 27)): "13",
}
# =====================

os.makedirs(os.path.dirname(temp_video) or ".", exist_ok=True)
os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)

image_files = [
    f
    for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic"))
]

image_files.sort(key=get_image_date)
if len(image_files) == 0:
    raise Exception("No images found!")

font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)


def get_real_date(path, img):
    """Get the actual date the photo was taken: EXIF first, then file mtime."""
    try:
        exif = img.getexif()
        if exif:
            # DateTime (main IFD)
            if 306 in exif:
                return datetime.strptime(exif[306], "%Y:%m:%d %H:%M:%S")
            # DateTimeOriginal / DateTimeDigitized (Exif sub-IFD, tag 0x8769)
            exif_ifd = exif.get_ifd(0x8769)
            for tag in (36867, 36868):
                if tag in exif_ifd:
                    return datetime.strptime(exif_ifd[tag], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    # Fall back to the file's real modification time
    return datetime.fromtimestamp(os.path.getmtime(path))


def process_image(img_name):
    """Decode, crop/resize, and stamp one image. Runs in a worker thread."""
    path = os.path.join(image_folder, img_name)
    try:
        img = Image.open(path)
        timestamp_dt = get_real_date(path, img)
        timestamp = timestamp_dt.strftime("%d.%m.%Y  %H:%M")

        img = img.convert("RGB")
        # Fill frame while preserving aspect ratio
        canvas = ImageOps.fit(img, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(canvas)

        # draw timestamp
        draw_text(
            draw,
            timestamp,
            font,
            WIDTH,
            HEIGHT,
            text_color=text_color,
            position="bottom-right",
        )

        # add current class
        current_class = get_current_class(timestamp_dt)
        draw_text(
            draw,
            f"{current_class}.",
            font,
            WIDTH,
            HEIGHT,
            text_color=text_color,
            position="top-left",
        )

        frame = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        return img_name, frame, None
    except Exception as e:
        return img_name, None, str(e)


def get_current_class(timestamp_dt: datetime) -> str | None:
    for (start, end), class_name in school_year_dict.items():
        if start <= timestamp_dt.date() <= end:
            return class_name
    return None


def draw_text(
    draw,
    text,
    font,
    img_width,
    img_height,
    margin=20,
    text_color=(255, 255, 255),
    stroke_width=2,
    stroke_color=(0, 0, 0),
    position="bottom-right",
):
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    if position == "bottom-right":
        x = img_width - text_width - margin
        y = img_height - text_height - margin
    elif position == "bottom-left":
        x = margin
        y = img_height - text_height - margin
    elif position == "top-right":
        x = img_width - text_width - margin
        y = margin
    elif position == "top-left":
        x = margin
        y = margin
    else:
        raise ValueError("'top-left', 'top-right', 'bottom-left', 'bottom-right'")

    draw.text(
        (x, y),
        text,
        font=font,
        fill=text_color,
        stroke_width=stroke_width,
        stroke_fill=stroke_color,
    )

    return draw


frame_count = max(1, round(seconds_per_image * fps))
worker_count = min(8, os.cpu_count() or 4)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(temp_video, fourcc, fps, (WIDTH, HEIGHT))
if not video_writer.isOpened():
    raise Exception(
        f"Could not open video writer for '{temp_video}'. "
        "Check that the output folder exists and is writable."
    )

written = 0
with ThreadPoolExecutor(max_workers=worker_count) as executor:
    for i, (img_name, frame, error) in enumerate(
        executor.map(process_image, image_files)
    ):
        if error:
            print(f"{i + 1}/{len(image_files)} - Skipping {img_name}: {error}")
            continue
        print(f"{i + 1}/{len(image_files)} - {img_name}")
        for _ in range(frame_count):
            video_writer.write(frame)
        written += 1

video_writer.release()

if written == 0:
    raise Exception("No frames were written — every image failed to process.")
print(f"Wrote {written}/{len(image_files)} images to video.")

# add audio
video_input = ffmpeg.input(temp_video)
audio_input = ffmpeg.input(music_file)

output = ffmpeg.output(
    video_input.video,
    audio_input.audio,
    output_video,
    vcodec="copy",
    acodec="aac",
    shortest=None,
)
output = output.overwrite_output()

# Run the command
ffmpeg.run(output)

print("Done!")
