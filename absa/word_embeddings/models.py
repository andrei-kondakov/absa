import os

from django.conf import settings
from django.contrib.postgres.fields import JSONField
from django.core.files.storage import FileSystemStorage
from django.db import models

models_dir = os.path.join(settings.ABSA_DIR, 'models')
word2vec_fs = FileSystemStorage(location=os.path.join(models_dir, 'w2v'))


class Word2Vec(models.Model):
    config = JSONField(blank=True, null=True)
    corpus = models.TextField(blank=True)
    file = models.FileField(storage=word2vec_fs)
    created_at = models.DateTimeField(auto_now_add=True)
    available = models.BooleanField(default=False)

    class Meta:
        verbose_name = 'Word2Vec'
        verbose_name_plural = 'Word2Vec models'

    def __str__(self):
        return f'{self.id}'
