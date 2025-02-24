import cv2 as cv
import numpy as np
import load_model
from typing import Tuple


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
    Regresar la posición del objeto en row, column, width y height
    """
    img = cv.imread(img)
    height, width, _ = img.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(
        imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)
    cropped_image = imgray[min_y:max_y, min_x:max_x]
    new_img = resize_with_pad(cropped_image, (28, 28))
    cv.imwrite("trazo.jpg", new_img)
    return new_img


def preprocess_image(image):
    # Normalizar los valores entre 0 y 1
    image = image / 255.0
    # Redimensionar para añadir el canal (1)
    image = image.reshape(1, 28, 28, 1)
    return image


def resize_with_pad(
    image: np.array,
    new_shape: Tuple[int, int],
    padding_color: Tuple[int] = (0, 0, 0),
) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape)) / max(original_shape)
    new_size = tuple([int(x * ratio * 0.7) for x in original_shape])
    image = cv.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    image = cv.copyMakeBorder(
        image, top, bottom, left, right, cv.BORDER_CONSTANT, value=padding_color
    )
    return image
