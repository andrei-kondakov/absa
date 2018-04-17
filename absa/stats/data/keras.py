from django.db.models import Max, Q

import numpy as np
from data.models import SentenceBatch, Task
from keras_ml.models import TrainSession
from keras_ml.utils import f1_score as f1


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
        Q(batch__task__aspect_entity='') | Q(batch__task__aspect_attribute='')
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


def aspect_detection_stats_2():
    def get_session(entity, attribute):
        return TrainSession.objects.filter(
            batch__task__type=Task.Type.ASPECT_DETECTION,
            batch__task__aspect_entity=entity,
            batch__task__aspect_attribute=attribute,
            f1_score__isnull=False
        ).order_by('-f1_score').first()

    def get_f1_score(entity_session, attr_session):
        batch = SentenceBatch.objects.get(
            task__type=Task.Type.ASPECT_DETECTION,
            task__aspect_entity=entity_session.batch.task.aspect_entity,
            task__aspect_attribute=attr_session.batch.task.aspect_attribute
        )

        if not batch:
            return

        entity_y_pred = entity_session.y_pred
        attr_y_pred = attr_session.y_pred

        assert len(entity_y_pred) == len(attr_y_pred)

        y_pred = []
        for i in range(len(entity_y_pred)):
            if entity_y_pred[i][0] == attr_y_pred[i][0] and entity_y_pred[i][0] == 1.0:
                y_pred.append(1)
            else:
                y_pred.append(0)

        return f1(y_true=batch.y_test, y_pred=y_pred)

    aspects = []

    ambience_general = get_session(entity='AMBIENCE', attribute='GENERAL')
    if ambience_general:
        aspects.append({
            'entity': ambience_general.batch.task.aspect_entity,
            'attribute': ambience_general.batch.task.aspect_attribute,
            'f1_score': ambience_general.f1_score
        })

    drinks = get_session(entity='DRINKS', attribute='')
    prices = get_session(entity='', attribute='PRICES')
    style_opts = get_session(entity='', attribute='STYLE_OPTIONS')
    quality = get_session(entity='', attribute='QUALITY')

    if drinks:
        if prices:
            f1_score = get_f1_score(drinks, prices)
            if not f1_score is None:
                aspects.append({
                    'entity': drinks.batch.task.aspect_entity,
                    'attribute': prices.batch.task.aspect_attribute,
                    'f1_score': f1_score
                })
        if quality:
            f1_score = get_f1_score(drinks, quality)
            if not f1_score is None:
                aspects.append({
                    'entity': drinks.batch.task.aspect_entity,
                    'attribute': quality.batch.task.aspect_attribute,
                    'f1_score': f1_score
                })
        if style_opts:
            f1_score = get_f1_score(drinks, style_opts)
            if not f1_score is None:
                aspects.append({
                    'entity': drinks.batch.task.aspect_entity,
                    'attribute': style_opts.batch.task.aspect_attribute,
                    'f1_score': f1_score
                })

    food = get_session(entity='FOOD', attribute='')
    if food:
        if prices:
            f1_score = get_f1_score(food, prices)
            if not f1_score is None:
                aspects.append({
                    'entity': food.batch.task.aspect_entity,
                    'attribute': prices.batch.task.aspect_attribute,
                    'f1_score': f1_score
                })
        if quality:
            f1_score = get_f1_score(food, quality)
            if not f1_score is None:
                aspects.append({
                    'entity': food.batch.task.aspect_entity,
                    'attribute': quality.batch.task.aspect_attribute,
                    'f1_score': f1_score
                })
        if style_opts:
            f1_score = get_f1_score(food, style_opts)
            if not f1_score is None:
                aspects.append({
                    'entity': food.batch.task.aspect_entity,
                    'attribute': style_opts.batch.task.aspect_attribute,
                    'f1_score': f1_score
                })

    location_general = get_session(entity='LOCATION', attribute='GENERAL')
    if location_general:
        aspects.append({
            'entity': location_general.batch.task.aspect_entity,
            'attribute': location_general.batch.task.aspect_attribute,
            'f1_score': location_general.f1_score
        })

    restaurant_general = get_session(entity='RESTAURANT', attribute='GENERAL')
    if restaurant_general:
        aspects.append({
            'entity': restaurant_general.batch.task.aspect_entity,
            'attribute': restaurant_general.batch.task.aspect_attribute,
            'f1_score': restaurant_general.f1_score
        })

    restaurant = get_session(entity='RESTAURANT', attribute='')
    miscellaneous = get_session(entity='', attribute='MISCELLANEOUS')
    if restaurant:
        if miscellaneous:
            f1_score = get_f1_score(restaurant, miscellaneous)
            if not f1_score is None:
                aspects.append({
                    'entity': restaurant.batch.task.aspect_entity,
                    'attribute': miscellaneous.batch.task.aspect_attribute,
                    'f1_score': f1_score
                })
        if prices:
            f1_score = get_f1_score(restaurant, prices)
            if not f1_score is None:
                aspects.append({
                    'entity': restaurant.batch.task.aspect_entity,
                    'attribute': prices.batch.task.aspect_attribute,
                    'f1_score': f1_score
                })

    service_general = get_session(entity='SERVICE', attribute='GENERAL')
    if service_general:
        aspects.append({
            'entity': service_general.batch.task.aspect_entity,
            'attribute': service_general.batch.task.aspect_attribute,
            'f1_score': service_general.f1_score
        })

    f1_macro = np.mean([x['f1_score'] for x in aspects])

    return {
        'aspects': aspects,
        'f1_macro': f1_macro
    }


def keras_stats():
    return {
        'aspect_detection_stats': aspect_detection_stats(),
        'aspect_detection_stats_2': aspect_detection_stats_2(),
        'session_stats': session_stats()
    }
