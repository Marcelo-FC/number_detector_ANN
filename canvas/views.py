from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.http import HttpResponse
from django.contrib.auth import authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout
from django.forms.models import model_to_dict


from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.cache import never_cache
from django.contrib.auth.forms import UserCreationForm
from datetime import date, datetime
from django.core.serializers import serialize 
from django.contrib.auth.hashers import make_password, check_password
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.db.models import Q

import base64
import io
import json
from PIL import Image
from django.http import JsonResponse
from django.shortcuts import render
from tools import object_crop, preprocess_image
import load_model
from PIL import ImageOps
from PIL import ImageEnhance

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
from tools import object_crop, preprocess_image
import load_model
import os
from django.conf import settings
# Create your views here.
@csrf_exempt
def home(request):
    return render(request,'canvas.html')


@csrf_exempt
def process_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        # Guardar la imagen en el servidor
        image_file = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, 'trazo.jpg')
        with open(image_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Procesar la imagen usando tools.py
        img = object_crop(image_path)  # tools.py espera una ruta de archivo
        img = preprocess_image(img)
        rec_char = load_model.predict_digit(img)

        # Devolver la respuesta
        return JsonResponse({'character': rec_char})
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)