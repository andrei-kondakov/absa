from django.db.models import Max, Q

import numpy as np
from data.models import Task
from keras_ml.models import TrainSession


def session_stats():
    return {
        'all': TrainSession.objects.count(),
        'trained': TrainSession.objects.exclude(y_pred__isnull=True).count(),
        'evaluated': TrainSession.objects.exclude(
            Q(precision__isnull=True) | Q(recall__isnull=True) | Q(f1_score__isnull=True)
        ).count()
    }


def aspect_detection_stats():
    raw_stats = TrainSession.objects.filter(
        batch__task__type=Task.Type.ASPECT_DETECTION,
        f1_score__isnull=False
    ).exclude(
        Q (batch__task__aspect_entity='') | Q(batch__task__aspect_attribute='')
    ).values(
        'batch__task__aspect_entity',
        'batch__task__aspect_attribute'
    ).annotate(
        f1_score=Max('f1_score')
    ).order_by(
        'batch__task__aspect_entity',
        'batch__task__aspect_attribute'
    )

    aspects = []
    for item in raw_stats:
        aspects.append({
            'entity': item['batch__task__aspect_entity'],
            'attribute': item['batch__task__aspect_attribute'],
            'f1_score': item['f1_score']
        })

    f1_macro = np.mean([x['f1_score'] for x in aspects])
    return {
        'aspects': aspects,
        'f1_macro': f1_macro
    }


def keras_stats():
    return {
        'aspect_detection_stats': aspect_detection_stats(),
        'session_stats': session_stats()
    }
