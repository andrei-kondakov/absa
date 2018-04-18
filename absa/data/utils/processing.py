from django.db.models import Q

from data.utils.preprocessing import preprocess
from word_embeddings.utils import word2vec_indexes_v1


class Processing:
    NO_PROCESSING = -1
    W2V_INDEXES_1 = 1

    choices = (
        (NO_PROCESSING, 'No processing'),
        (W2V_INDEXES_1, 'Word2vec indexes v1')
    )


def get_ys(sentences, target_category):
    ys = []
    for sentence in sentences:
        y = int(any([target_category in category for category in sentence.categories]))
        ys.append(y)
    return ys

def get_ys_polarity(sentences, target_polarity):
    ys = []
    for sentence in sentences:
        y = target_polarity == sentence.polarities[0].upper()
        ys.append(y)
    return ys


def process(sentence_batch):
    from data.models import Task, TestSentence, TrainSentence

    preprocessing = sentence_batch.preprocessing
    processing = sentence_batch.processing
    task = sentence_batch.task

    if processing == Processing.NO_PROCESSING:
        return sentence_batch

    if task.type == Task.Type.ASPECT_DETECTION:
        if processing == Processing.W2V_INDEXES_1:
            entity = task.aspect_entity or ''
            attribute = task.aspect_attribute or ''

            category = f'{entity.upper()}#{attribute.upper()}'

            train_sentences = TrainSentence.objects.filter(
                out_of_scope=False
            )

            test_sentences = TestSentence.objects.filter(
                out_of_scope=False
            )

            if not len(train_sentences) or not len(test_sentences):
                raise ValueError('You should loaded datasets')

            x_train_raw = [x.text for x in train_sentences]
            x_train_preproc = preprocess(x_train_raw, preprocessing)
            y_train_raw = [','.join(x.categories) for x in train_sentences]
            x_train = word2vec_indexes_v1(sentence_batch.w2v_model, x_train_preproc)
            y_train = get_ys(train_sentences, category)

            x_test_raw = [x.text for x in test_sentences]
            x_test_preproc = preprocess(x_test_raw, preprocessing)
            y_test_raw = [','.join(x.categories) for x in test_sentences]
            x_test = word2vec_indexes_v1(sentence_batch.w2v_model, x_test_preproc)
            y_test = get_ys(test_sentences, category)

            sentence_batch.x_train_raw = x_train_raw
            sentence_batch.x_train_preproc = x_train_preproc
            sentence_batch.y_train_raw = y_train_raw
            sentence_batch.x_train = x_train
            sentence_batch.y_train = y_train

            sentence_batch.x_test_raw = x_test_raw
            sentence_batch.x_test_preproc = x_test_preproc
            sentence_batch.y_test_raw = y_test_raw
            sentence_batch.x_test = x_test
            sentence_batch.y_test = y_test
            return sentence_batch

    if task.type == Task.Type.POLARITY_DETECTION:
        if processing == Processing.W2V_INDEXES_1:
            if not task.polarity :
                raise ValueError

            polarity = task.polarity.upper()

            train_sentences = TrainSentence.objects.filter(
                out_of_scope=False,
                polarities__len=1
            ).exclude(
                polarities=['conflict']
            )

            test_sentences = TestSentence.objects.filter(
                out_of_scope=False,
                polarities__len=1
            ).exclude(
                polarities=['conflict']
            )

            x_train_raw = [x.text for x in train_sentences]
            x_train_preproc = preprocess(x_train_raw, preprocessing)
            y_train_raw = [x.polarities for x in train_sentences]
            x_train = word2vec_indexes_v1(sentence_batch.w2v_model, x_train_preproc)
            y_train = get_ys_polarity(train_sentences, polarity)

            x_test_raw = [x.text for x in test_sentences]
            x_test_preproc = preprocess(x_test_raw, preprocessing)
            y_test_raw = [x.polarities for x in test_sentences]
            x_test = word2vec_indexes_v1(sentence_batch.w2v_model, x_test_preproc)
            y_test = get_ys_polarity(test_sentences, polarity)

            sentence_batch.x_train_raw = x_train_raw
            sentence_batch.x_train_preproc = x_train_preproc
            sentence_batch.y_train_raw = y_train_raw
            sentence_batch.x_train = x_train
            sentence_batch.y_train = y_train

            sentence_batch.x_test_raw = x_test_raw
            sentence_batch.x_test_preproc = x_test_preproc
            sentence_batch.y_test_raw = y_test_raw
            sentence_batch.x_test = x_test
            sentence_batch.y_test = y_test
            return sentence_batch


    raise NotImplementedError
