from collections import defaultdict

from django.shortcuts import get_object_or_404, render

from data.models import Task
from keras_ml.models import TrainSession


def sessions(request):
    data = {
        'tasks': []
    }
    tasks = Task.objects.all()

    for task in tasks:
        train_sessions = defaultdict(list)
        for session in TrainSession.objects.filter(batch__task_id=task.id):
            train_sessions[session.model_id].append(session)

        data['tasks'].append(
            {
                'task': task,
                'sessions': dict(train_sessions)
            }
        )

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
