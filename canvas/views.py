# 📌 Importaciones de Django necesarias para manejar autenticación, vistas y respuestas JSON
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib.auth import authenticate, login as auth_login, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.decorators import login_required
from django.core.serializers import serialize
from django.contrib.auth.hashers import make_password, check_password
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.db.models import Q

# 📌 Importaciones adicionales necesarias para la gestión de imágenes
import base64
import io
import json
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
import os

# 📌 Importaciones específicas del proyecto
from tools import object_crop, preprocess_image  # Funciones para procesar imágenes
import load_model  # Módulo que maneja la inferencia con TFLite
from django.conf import settings

# 📌 Decoradores de Django para seguridad y control de vistas
from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.cache import never_cache

# 📌 Vista principal (Renderiza la página principal con el lienzo para dibujar números)
@csrf_exempt
def home(request):
    """
    Renderiza la página principal donde el usuario puede dibujar un número 
    y enviarlo para reconocimiento.
    """
    return render(request, 'canvas.html')

# 📌 Vista para procesar imágenes enviadas por el usuario
@csrf_exempt
def process_image(request):
    """
    Maneja el procesamiento de la imagen enviada por el usuario.
    1️⃣ Guarda la imagen recibida en el servidor.
    2️⃣ Preprocesa la imagen para ajustarla al formato del modelo.
    3️⃣ Realiza la predicción con el modelo TFLite cargado.
    4️⃣ Devuelve el resultado de la predicción en formato JSON.

    Retorna:
        - Un JSON con la predicción del carácter detectado.
        - En caso de error, retorna un mensaje de error.
    """
    if request.method == 'POST' and 'image' in request.FILES:
        # 📌 Guardar la imagen en el servidor
        image_file = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, 'trazo.jpg')  # Ruta donde se guarda la imagen

        with open(image_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # 📌 Procesar la imagen para adaptarla al modelo de predicción
        img = object_crop(image_path)  # Recorta y ajusta el dígito en la imagen
        img = preprocess_image(img)  # Normaliza y centra la imagen antes de enviarla al modelo

        # 📌 Realizar la predicción con el modelo TFLite
        rec_char = load_model.predict_digit(img)

        # 📌 Devolver la respuesta en formato JSON con el carácter reconocido
        return JsonResponse({'character': rec_char})
    
    else:
        # 📌 Si la solicitud no es válida, devolver un mensaje de error
        return JsonResponse({'error': 'Invalid request'}, status=400)
