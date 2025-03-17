import cv2 as cv
import numpy as np
from typing import Tuple
import os

# 📂 Crear un directorio de depuración si no existe
# Este directorio almacenará imágenes intermedias para depuración.
DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)


def show_image(img):
    """
    Muestra una imagen en una ventana emergente.
    
    Si la imagen no está en formato uint8 o complejo128, la normaliza a valores entre 0 y 255.
    """
    i_types = ["uint8", "complex128"]
    if img.dtype not in i_types:
        img = normalize_img(img, 0, 255)
    cv.imshow("Image", img)
    cv.waitKey()
    cv.destroyAllWindows()


def normalize_img(img, alpha: int, beta: int):
    """
    Normaliza la imagen a un rango de valores entre alpha y beta.
    
    Esta función es útil para garantizar que los valores de píxeles estén dentro de un rango
    adecuado para su visualización o procesamiento.
    
    Parámetros:
    - img: Imagen en escala de grises o en otro formato.
    - alpha: Límite inferior del rango de normalización.
    - beta: Límite superior del rango de normalización.
    
    Retorna:
    - Imagen normalizada en uint8.
    """
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
    Recorta el dígito escrito a mano de la imagen y lo redimensiona con padding.

    - Convierte la imagen a escala de grises.
    - Encuentra los contornos del número escrito.
    - Determina la región más pequeña que contiene el número.
    - Recorta la imagen para enfocarse solo en el dígito.
    - Guarda la imagen recortada y ajustada para depuración.

    Parámetros:
    - img: Ruta del archivo de la imagen original.

    Retorna:
    - Imagen en escala de grises de tamaño ajustado.
    """
    img = cv.imread(img)  # Carga la imagen original
    height, width, _ = img.shape  # Obtiene las dimensiones
    min_x, min_y = width, height
    max_x = max_y = 0

    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convierte a escala de grises

    # Detectar contornos en la imagen
    contours, _ = cv.findContours(
        imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    
    # Encuentra los límites del objeto más grande (el número escrito)
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)

    # Recortar la imagen para aislar el número
    cropped_image = imgray[min_y:max_y, min_x:max_x]

    # Guardar imagen recortada para depuración
    cv.imwrite(os.path.join(DEBUG_DIR, "debug_cropped.jpg"), cropped_image)
    print("✅ Imagen recortada guardada en debug/debug_cropped.jpg")

    # Redimensionar la imagen con padding para ajustarla a 28x28
    new_img = resize_with_pad(cropped_image, (28, 28))

    # Guardar imagen redimensionada para depuración
    cv.imwrite(os.path.join(DEBUG_DIR, "debug_resized.jpg"), new_img)
    print("✅ Imagen redimensionada guardada en debug/debug_resized.jpg")

    # Guardar la imagen final
    cv.imwrite("trazo.jpg", new_img)

    return new_img


def preprocess_image(image):
    """
    Normaliza la imagen y aplica preprocesamiento adicional para mejorar la predicción.

    - Convierte los valores de la imagen a un rango entre 0 y 1.
    - Opcionalmente, engrosa las líneas del número con un operador de dilatación.
    - Centra la imagen usando el centro de masa del número.
    - Reformatea la imagen para que sea compatible con la entrada del modelo.

    Parámetros:
    - image: Imagen en escala de grises con el número escrito.

    Retorna:
    - Imagen preprocesada lista para el modelo.
    """
    # Normalizar los valores de píxeles entre 0 y 1
    image = image / 255.0

    # Aplicar dilatación para engrosar el número
    kernel = np.ones((2, 2), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)

    # Centrar el número en la imagen
    image = shift_to_center(image)

    # Reformatear la imagen para el modelo
    final_image = image.reshape(1, 28, 28)

    # Guardar la imagen preprocesada para depuración
    debug_image = (image * 255).astype(np.uint8)
    cv.imwrite(os.path.join(DEBUG_DIR, "debug_final_preprocessed.jpg"), debug_image)
    print("✅ Imagen preprocesada guardada en debug/debug_final_preprocessed.jpg")

    return final_image


def resize_with_pad(image: np.array, new_shape: Tuple[int, int], padding_color: Tuple[int] = (0, 0, 0)) -> np.array:
    """
    Redimensiona la imagen manteniendo la relación de aspecto y aplica relleno para ajustarla a 28x28.

    - Reduce la imagen de manera proporcional.
    - Agrega bordes negros para ajustarla al tamaño deseado.

    Parámetros:
    - image: Imagen de entrada en escala de grises.
    - new_shape: Dimensiones finales de la imagen (28x28).
    - padding_color: Color de relleno (por defecto, negro).

    Retorna:
    - Imagen ajustada con relleno.
    """
    h, w = image.shape
    target_size = 20  # Reservar espacio para ajustes posteriores

    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)

    # Calcular relleno
    pad_h = (new_shape[0] - new_h) // 2
    pad_w = (new_shape[1] - new_w) // 2

    padded = cv.copyMakeBorder(
        resized, pad_h, new_shape[0] - new_h - pad_h, pad_w, new_shape[1] - new_w - pad_w,
        cv.BORDER_CONSTANT, value=padding_color
    )

    return padded


def shift_to_center(image, canvas_size=40, final_size=28):
    """
    Centra el número en la imagen utilizando un lienzo más grande para evitar recortes.

    - Ubica la imagen en un lienzo de 40x40 píxeles.
    - Calcula el centro de masa del número.
    - Aplica un desplazamiento para alinear el centro del número con el centro de la imagen.
    - Recorta la imagen nuevamente a 28x28.

    Parámetros:
    - image: Imagen en escala de grises del número.
    - canvas_size: Tamaño del lienzo intermedio (40x40 por defecto).
    - final_size: Tamaño final después del recorte (28x28).

    Retorna:
    - Imagen ajustada y centrada de 28x28 píxeles.
    """
    from scipy.ndimage import center_of_mass

    canvas = np.zeros((canvas_size, canvas_size), dtype=image.dtype)
    x_offset = (canvas_size - image.shape[1]) // 2
    y_offset = (canvas_size - image.shape[0]) // 2
    canvas[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image

    cy, cx = center_of_mass(canvas)
    shiftx = np.round(canvas_size / 2.0 - cx).astype(int)
    shifty = np.round(canvas_size / 2.0 - cy).astype(int)

    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    shifted_canvas = cv.warpAffine(canvas, M, (canvas_size, canvas_size), borderValue=0)

    final_image = shifted_canvas[6:34, 6:34]

    return final_image
