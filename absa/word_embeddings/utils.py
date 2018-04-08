import gensim
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# dummy caches of loaded w2v models
_cached_models = {}


def load_w2v_model(w2v_model):
    if w2v_model.id in _cached_models:
        print(f'Loading w2v model using cache')
        return _cached_models[w2v_model.id]

    w2v_model_path = w2v_model.file.path
    print(f'Loading w2v model from {w2v_model_path}')
    model = gensim.models.Word2Vec.load(w2v_model_path)
    _cached_models[w2v_model.id] = model
    return model


def create_embedding_matrix(w2v_loaded_model):
    word_count = len(w2v_loaded_model.wv.vocab)
    matrix = np.zeros((word_count, w2v_loaded_model.vector_size))

    for word, info in w2v_loaded_model.wv.vocab.items():
        index = info.index
        matrix[index] = w2v_loaded_model.wv.syn0[index]
    return matrix


def word2vec_indexes_v1(w2v_model, sentences):
    model = load_w2v_model(w2v_model)

    X = []
    for sentence in sentences:
        word_indexes = []

        for word in sentence.split():
            word_index = model.wv.vocab[word].index if word in model.wv.vocab else 0
            word_indexes.append(word_index)

        X.append(word_indexes)

    return pad_sequences(X, maxlen=60).tolist()
