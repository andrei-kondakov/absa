import os

from django.conf import settings

KERAS_MODELS_DIR = os.path.join(settings.ABSA_DIR, 'models', 'keras')
