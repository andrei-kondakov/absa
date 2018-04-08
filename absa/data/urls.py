from django.urls import path

import data.views

urlpatterns = [
    path('batch/<int:batch_id>', data.views.batch_detail, name='batch_detail')
]
