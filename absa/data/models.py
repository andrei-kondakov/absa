from django.contrib.postgres.fields import ArrayField
from django.db import models

from data.utils.preprocessing import Preprocessing
from data.utils.processing import Processing, process
from word_embeddings.models import Word2Vec


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
        db_table = 'data_train_sentences'


class TestSentence(Sentence):
    class Meta:
        db_table = 'data_test_sentences'


class Task(models.Model):
    class Type:
        ASPECT_DETECTION = 1
        POLARITY_DETECTION = 2

        choices = (
            (ASPECT_DETECTION, 'Aspect detection'),
            (POLARITY_DETECTION, 'Polarity detection')
        )

    type = models.SmallIntegerField(choices=Type.choices)
    aspect_entity = models.TextField(blank=True)
    aspect_attribute = models.TextField(blank=True)
    polarity = models.TextField(blank=True)

    class Meta:
        unique_together = ('type', 'aspect_entity', 'aspect_attribute', 'polarity')

    def __str__(self):
        task_str = self.get_type_display()
        if self.type == Task.Type.ASPECT_DETECTION:
            if self.aspect_entity:
                task_str += f', entity: {self.aspect_entity}'
            if self.aspect_attribute:
                task_str += f', attribute: {self.aspect_attribute}'
        if self.type == Task.Type.POLARITY_DETECTION:
            if self.polarity:
                task_str += f', polarity: {self.polarity}'
        return task_str


class UnknownWord(models.Model):
    class MorphAnalyzer:
        UNKNOWN = -1
        PYMORPHY2 = 1

        choices = (
            (UNKNOWN, 'Unknown'),
            (PYMORPHY2, 'Pymorphy2')
        )

    word = models.TextField()
    morph_analyzer = models.TextField(choices=MorphAnalyzer.choices, default=MorphAnalyzer.UNKNOWN)

    class Meta:
        db_table = 'data_unknown_word'


class SentenceBatch(models.Model):
    task = models.ForeignKey(to=Task, on_delete=models.PROTECT)
    preprocessing = models.SmallIntegerField(choices=Preprocessing.choices)
    processing = models.SmallIntegerField(choices=Processing.choices)

    w2v_model = models.ForeignKey(to=Word2Vec, on_delete=models.PROTECT, blank=True, null=True)

    x_train_raw = ArrayField(base_field=models.TextField(), blank=True)
    x_train_preproc = ArrayField(base_field=models.TextField(), blank=True)
    y_train_raw = ArrayField(base_field=models.TextField(), blank=True)

    x_train = ArrayField(
        base_field=ArrayField(base_field=models.IntegerField()),
        blank=True
    )
    y_train = ArrayField(base_field=models.IntegerField(), blank=True)

    x_test_raw = ArrayField(base_field=models.TextField(), blank=True)
    x_test_preproc = ArrayField(base_field=models.TextField(), blank=True)
    y_test_raw = ArrayField(base_field=models.TextField(), blank=True)

    x_test = ArrayField(
        base_field=ArrayField(base_field=models.IntegerField()),
        blank=True
    )
    y_test = ArrayField(base_field=models.IntegerField(), blank=True)

    class Meta:
        db_table = 'data_sentence_batch'

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        if not self.pk:
            process(self)

        super().save(force_insert, force_update, using, update_fields)

    def __str__(self):
        return f'#{self.id}, for task: {self.task}'
