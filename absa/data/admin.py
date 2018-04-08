from django.contrib import admin

from data.models import SentenceBatch, Task


@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    list_display = ('id', 'type', 'aspect_entity', 'aspect_attribute', 'polarity')
    ordering = ('id',)


@admin.register(SentenceBatch)
class SentenceBatchAdmin(admin.ModelAdmin):
    list_display = ('id', 'task', 'preprocessing', 'processing', 'w2v_model')
    ordering = ('id',)
