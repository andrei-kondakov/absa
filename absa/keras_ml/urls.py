from django.urls import path

import keras_ml.views

urlpatterns = [
    path('session/<int:session_id>', keras_ml.views.session_detail, name='session_detail')
]
