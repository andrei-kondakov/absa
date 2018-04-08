import re

from data.utils.morphology import pymorphy_normalize


class Preprocessing:
    NO_PREPROCESSING = -1
    PREPROCESSING_1 = 1

    choices = (
        (NO_PREPROCESSING, 'No preprocessing'),
        (PREPROCESSING_1, 'Preprocessing v1'),
    )


def strip_nonalnum_re(word):
    """Strip redundant symbols for give word.

    Example:
    >>> strip_nonalnum_re('abc,')
    'abc'
    >>> strip_nonalnum_re('.abc')
    'abc'
    """
    return re.sub(r"^\W+|\W+$", "", word)


def preprocess(raw_sentences, preprocessing):
    if preprocessing == Preprocessing.NO_PREPROCESSING:
        return raw_sentences

    if preprocessing == Preprocessing.PREPROCESSING_1:
        preprocessed_sentences = []
        for sentence in raw_sentences:
            sentence = sentence.lower()
            sentence = sentence \
                .replace('…', '') \
                .replace(',', ' , ') \
                .replace(':', ' : ')

            n_sentence = ''

            for word in sentence.split():
                word = strip_nonalnum_re(word)

                if len(word) == 0:
                    continue

                word_normal_form, tag = pymorphy_normalize(word)

                if tag and {'PNCT'} not in tag:
                    n_sentence += word_normal_form + ' '

            preprocessed_sentences.append(n_sentence.rstrip())

        return preprocessed_sentences

    raise ValueError(f'Unsupported processing: {preprocessing}')
