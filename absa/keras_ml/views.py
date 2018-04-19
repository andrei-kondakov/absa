import json
from collections import defaultdict

from django.shortcuts import get_object_or_404, render

from data.models import Task
from data.utils.preprocessing import Preprocessing
from data.utils.processing import Processing
from keras_ml.models import Model, TrainSession

task_type_label = dict(Task.Type.choices)
processing_label = dict(Processing.choices)
preprocessing_label = dict(Preprocessing.choices)


def sessions(request):
    data = {
        'train_sessions': defaultdict(list)
    }
    train_sessions = TrainSession.objects.filter(
        f1_score__isnull=False
    ).values(
        'id',
        'model_id',
        'batch__preprocessing',
        'batch__processing',
        'batch__task',
        'batch__task__type',
        'batch__task__aspect_attribute',
        'batch__task__aspect_entity',
        'batch__task__polarity',
        'f1_score'
    ).order_by('-id')

    for session in train_sessions:
        session['preprocessing'] = preprocessing_label[session['batch__preprocessing']]
        session['processing'] = processing_label[session['batch__processing']]
        task_desc = [
            task_type_label[session['batch__task__type']],
            session['batch__task__aspect_entity'],
            session['batch__task__aspect_attribute'],
            session['batch__task__polarity']
        ]
        task = ', '.join(filter(None, task_desc))
        data['train_sessions'][task].append(session)

    # FIXME tempory solution
    data['train_sessions'] = dict(sorted(dict(data['train_sessions']).items()))

    return render(request, 'keras_ml/sessions.html', data)


def session_detail(request, session_id):
    session = get_object_or_404(TrainSession, id=session_id)

    error_sentences = []

    for i in range(len(session.y_pred)):
        y_pred = int(session.y_pred[i][0])
        y_true = int(session.batch.y_test[i])

        if y_pred != y_true:
            error_sentences.append({
                'raw': session.batch.x_test_raw[i],
                'preprocessing': session.batch.x_test_preproc[i],
                'x': session.batch.x_test[i],
                'y_true': y_true,
                'y_pred': y_pred
            })

    data = {
        'session': session,
        'error_sentences': error_sentences
    }

    return render(request, 'keras_ml/session_detail.html', data)


def model_detail(request, model_id):
    model = get_object_or_404(Model, id=model_id)

    data = {
        'model_id': model.id,
        'model_config': json.dumps(model.config, indent=4),
        'model_compile_opts': json.dumps(model.compile_opts, indent=4)
    }

    return render(request, 'keras_ml/model_detail.html', data)
