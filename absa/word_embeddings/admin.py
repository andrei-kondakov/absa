from django.contrib import admin

from word_embeddings.models import Word2Vec


@admin.register(Word2Vec)
class Word2VecAdmin(admin.ModelAdmin):
    list_display = ('id', 'config', 'corpus', 'file', 'created_at', 'available')
