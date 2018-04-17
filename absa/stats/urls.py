from django.urls import path

import stats.views

urlpatterns = [
    path('', stats.views.dashboard, name='dashboard')
]
