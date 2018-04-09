import logging
import os
from datetime import datetime
from pathlib import Path

from celery import shared_task

import numpy as np
from keras.layers.embeddings import Embedding
from keras.models import model_from_config
from keras_ml.constants import KERAS_MODELS_DIR
from keras_ml.models import TrainSession
from keras_ml.utils import TimingCallback, get_metrics
from word_embeddings.utils import create_embedding_matrix, load_w2v_model

logger = logging.getLogger('absa')


@shared_task
def train(session_id):
    session = TrainSession.objects.get(id=session_id)

    if len(session.y_pred):
        return

    model = model_from_config(session.model.config)

    # Check if model has Embedding layer and if there is such a layer we need to set weights from word_embeddings model
    embedding_layer = None

    # Embedding layer must be the first, but we add an iteration for the general case
    for layer in model.layers:
        if isinstance(layer, Embedding):
            embedding_layer = layer

    if embedding_layer:
        if session.batch.w2v_model:
            w2v_model = load_w2v_model(session.batch.w2v_model)
            embedding_layer.set_weights([create_embedding_matrix(w2v_model)])
        else:
            logger.warning(f'Model #{session.model.id} has embedding layer, but session.batch.w2v_model is null')

    model.compile(**session.model.compile_opts)
    timing_callback = TimingCallback()

    x_train = np.array(session.batch.x_train)
    y_train = np.array(session.batch.y_train)

    x_test = np.array(session.batch.x_test)

    history_obj = model.fit(x=x_train, y=y_train, callbacks=[timing_callback], **session.train_opts)

    # Save model
    now = datetime.now()
    model_dir = os.path.join(KERAS_MODELS_DIR, f'{now.day}-{now.month}-{now.year}')
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_dir, f'{session.id}.h5')
    model.save(model_path)

    session.y_pred = model.predict_classes(x_test).tolist()
    session.exec_time = timing_callback.logs
    session.history = history_obj.history
    session.model_filepath = model_path
    session.save()

    evaluate.delay(session.id)


@shared_task()
def evaluate(session_id):
    session = TrainSession.objects.get(id=session_id)

    if not session.evaluation is None:
        return

    session.evaluation = get_metrics(session.batch.y_test, session.y_pred)
    session.save()


@shared_task
def train_all():
    sessions = TrainSession.objects.filter(y_pred=[])
    for session in sessions:
        train.delay(session.id)


@shared_task()
def evaluate_all():
    sessions = TrainSession.objects.filter(evaluation__isnull=True)
    for session in sessions:
        evaluate.delay(session.id)
