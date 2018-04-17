from django.contrib import admin

from keras_ml.models import Model, TrainSession


@admin.register(Model)
class ModelAdmin(admin.ModelAdmin):
    list_display = ('config', 'compile_opts')
    ordering = ('id',)


@admin.register(TrainSession)
class TrainSessionAdmin(admin.ModelAdmin):
    list_display = ('model', 'batch', 'precision', 'recall', 'f1_score')
    ordering = ('id',)
