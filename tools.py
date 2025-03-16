import cv2 as cv
import numpy as np
from typing import Tuple
import os


# Create debug directory if it doesn't exist
DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)


def show_image(img):
    i_types = ["uint8", "complex128"]
    if img.dtype not in i_types:
        img = normalize_img(img, 0, 255)
    cv.imshow("Image", img)
    cv.waitKey()
    cv.destroyAllWindows()


def normalize_img(img, alpha: int, beta: int):
    n_img = cv.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv.NORM_MINMAX,
        dtype=cv.CV_8U,
    )
    return n_img


def object_crop(img):
    """
    Crop the handwritten digit from the image and resize it with padding.
    """
    img = cv.imread(img)
    height, width, _ = img.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contours, _ = cv.findContours(
        imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)
    cropped_image = imgray[min_y:max_y, min_x:max_x]

    # Save for debugging
    cv.imwrite(os.path.join(DEBUG_DIR, "debug_cropped.jpg"), cropped_image)
    print("✅ Cropped image saved to debug/debug_cropped.jpg")

    # Resize and pad properly
    new_img = resize_with_pad(cropped_image, (28, 28))

    # Save final resized image for debugging
    cv.imwrite(os.path.join(DEBUG_DIR, "debug_resized.jpg"), new_img)
    print("✅ Resized and padded image saved to debug/debug_resized.jpg")

    # Save final processed input
    cv.imwrite("trazo.jpg", new_img)

    return new_img


def preprocess_image(image):
    """
    Normalize and optionally thicken digit for prediction.
    """
    # Normalize to [0, 1]
    image = image / 255.0

    # Optional dilation to thicken digits
    kernel = np.ones((2, 2), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)

    # Center the digit more accurately
    image = shift_to_center(image)

    # Reshape for model input
    final_image = image.reshape(1, 28, 28)

    # ---------- SAVE FOR DEBUGGING ----------
    debug_image = (image * 255).astype(np.uint8)
    cv.imwrite(os.path.join(DEBUG_DIR, "debug_final_preprocessed.jpg"), debug_image)
    print("✅ Preprocessed image saved to debug/debug_final_preprocessed.jpg")
    # ----------------------------------------

    return final_image


def resize_with_pad(
    image: np.array,
    new_shape: Tuple[int, int],
    padding_color: Tuple[int] = (0, 0, 0),
) -> np.array:
    """
    Resize so that largest side is 20 pixels, then pad to 28x28.
    """
    h, w = image.shape
    target_size = 20  # Reserve space for shifting later

    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)

    # Padding
    pad_h = (new_shape[0] - new_h) // 2
    pad_w = (new_shape[1] - new_w) // 2

    padded = cv.copyMakeBorder(
        resized, pad_h, new_shape[0] - new_h - pad_h, pad_w, new_shape[1] - new_w - pad_w,
        cv.BORDER_CONSTANT, value=padding_color
    )

    return padded


def shift_to_center(image, canvas_size=40, final_size=28):
    """
    Shift digit to the center using a larger canvas to avoid clipping.
    """
    from scipy.ndimage import center_of_mass

    # Place image in larger canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=image.dtype)
    x_offset = (canvas_size - image.shape[1]) // 2
    y_offset = (canvas_size - image.shape[0]) // 2
    canvas[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image

    # Center of mass on canvas
    cy, cx = center_of_mass(canvas)
    shiftx = np.round(canvas_size / 2.0 - cx).astype(int)
    shifty = np.round(canvas_size / 2.0 - cy).astype(int)

    # Apply shift
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    shifted_canvas = cv.warpAffine(canvas, M, (canvas_size, canvas_size), borderValue=0)

    # Crop back to 28x28
    start_x = (canvas_size - final_size) // 2
    start_y = (canvas_size - final_size) // 2
    final_image = shifted_canvas[start_y:start_y + final_size, start_x:start_x + final_size]

    # ---------- Save for debugging ----------
    cv.imwrite(os.path.join(DEBUG_DIR, "debug_centered.jpg"), (final_image * 255).astype(np.uint8))
    print("✅ Centered image saved to debug/debug_centered.jpg")
    # ----------------------------------------

    return final_image
