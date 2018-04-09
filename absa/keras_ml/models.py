from django.contrib.postgres.fields import ArrayField, JSONField
from django.db import models

from data.models import SentenceBatch


class Model(models.Model):
    config = JSONField()
    compile_opts = JSONField()

    class Meta:
        unique_together = ('config', 'compile_opts')


class TrainSession(models.Model):
    model = models.ForeignKey(to=Model, on_delete=models.PROTECT)
    batch = models.ForeignKey(to=SentenceBatch, on_delete=models.PROTECT)
    train_opts = JSONField(null=True, blank=True)
    train_on_gpu = models.BooleanField(default=False)

    y_pred = ArrayField(
        base_field=ArrayField(base_field=models.FloatField()),
        null=True,
        blank=True
    )
    history = JSONField(null=True, blank=True)
    evaluation = JSONField(null=True, blank=True)
    exec_time = ArrayField(base_field=models.FloatField(), null=True, blank=True)
    model_filepath = models.FilePathField(blank=True, null=True)

    class Meta:
        db_table = 'keras_ml_train_session'

    def __str__(self):
        return f'#{self.id}, {self.model_id}, {self.batch.task}'
