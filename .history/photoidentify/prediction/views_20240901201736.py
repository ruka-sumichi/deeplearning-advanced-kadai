from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img、
from tensorflow.keras.preprocessing.image import 
from io import BytesIO
import os

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = img_array/243
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'model.h5')
            model = load_model(model_path)
            result = model.predict(img_array)


            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'prediction': prediction, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})