from django.contrib.postgres.fields import ArrayField, JSONField
from django.db import models


class Sentence(models.Model):
    sid = models.TextField(unique=True)
    text = models.TextField()
    out_of_scope = models.BooleanField()
    categories = ArrayField(base_field=models.TextField())
    polarities = ArrayField(base_field=models.TextField())

    class Meta:
        abstract = True
        indexes = [
            models.Index(fields=['out_of_scope'])
        ]

    def __str__(self):
        return f'{self.sid}, {self.text}'


class TrainSentence(Sentence):
    class Meta:
        db_table = 'train_sentences'


class TestSentence(Sentence):
    class Meta:
        db_table = 'test_sentences'
