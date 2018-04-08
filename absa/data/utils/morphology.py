# import logging

from django.core.cache import cache

import pymorphy2

# logger = logging.getLogger('absa')

morph = pymorphy2.MorphAnalyzer()


def pymorphy_normalize(word):
    """Return a word in normal form.

    Note: works only with the Russian language.
    """

    def normalize(word):
        if not morph.word_is_known(word):
            from data.models import UnknownWord

            UnknownWord.objects.get_or_create(
                word=word,
                morph_analyzer=UnknownWord.MorphAnalyzer.PYMORPHY2
            )
            # logger.error(f'Unknown word: {word}')

        word_parse = morph.parse(word)[0]
        normal_form = word_parse.normal_form
        tag = word_parse.tag
        return normal_form, tag

    get_normalized_word = lambda: normalize(word)
    return cache.get_or_set(f'pymorphy2:{word}', get_normalized_word, 60 * 60)
