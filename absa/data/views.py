from django.shortcuts import get_object_or_404, render

from data.models import SentenceBatch


def batch_detail(request, batch_id):
    batch = get_object_or_404(SentenceBatch, id=batch_id)

    train_sentences_table = {
        'rows': []
    }

    test_sentences_table = {
        'rows': []
    }

    for i in range(len(batch.x_train)):
        train_row = {
            'raw': batch.x_train_raw[i],
            'preprocessing': batch.x_train_preproc[i],
            'x': batch.x_train[i],
            'y_raw': batch.y_train_raw[i].replace(',', '<br>') or 'no category',
            'y': batch.y_train[i]
        }
        train_sentences_table['rows'].append(train_row)

    for i in range(len(batch.x_test)):
        test_row = {
            'raw': batch.x_test_raw[i],
            'preprocessing': batch.x_test_preproc[i],
            'x': batch.x_test[i],
            'y_raw': batch.y_test_raw[i].replace(',', '<br>') or 'no category',
            'y': batch.y_test[i]
        }
        test_sentences_table['rows'].append(test_row)

    data = {
        'batch': batch,
        'train_sentences_table': train_sentences_table,
        'test_sentences_table': test_sentences_table,
    }

    return render(request, 'data/batch_detail.html', data)
