from django.shortcuts import render

from stats.data.keras import keras_stats


def dashboard(request):
    data = {
        'keras': keras_stats()
    }
    print(data['keras']['polarity_detection_stats'])
    return render(request, 'stats/dashboard.html', data)
