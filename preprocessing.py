
import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image, size=32):
    # Convert to grayscale.
    image = image.convert("L")

    # Make the background white and the symbol dark.
    image = ImageOps.autocontrast(image)

    pixels = np.array(image)

    # Find dark pixels that likely belong to the symbol.
    mask = pixels < 200
    coordinates = np.argwhere(mask)

    # Return a blank white image if nothing was drawn.
    if len(coordinates) == 0:
        return np.ones(size * size, dtype=np.float32)

    # Crop around the symbol.
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)

    cropped = image.crop((
        x_min,
        y_min,
        x_max + 1,
        y_max + 1
    ))

    # Resize proportionally, leaving room for padding.
    cropped.thumbnail((size - 6, size - 6))

    # Centre the symbol on a white square.
    canvas = Image.new("L", (size, size), 255)
    x_offset = (size - cropped.width) // 2
    y_offset = (size - cropped.height) // 2

    canvas.paste(cropped, (x_offset, y_offset))

    # Convert to normalised pixel values.
    pixels = np.array(canvas, dtype=np.float32) / 255.0

    return pixels.flatten()
