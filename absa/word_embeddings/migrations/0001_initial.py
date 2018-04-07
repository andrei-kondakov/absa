# Generated by Django 2.0.4 on 2018-04-07 21:59

import django.contrib.postgres.fields.jsonb
import django.core.files.storage
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Word2Vec',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('config', django.contrib.postgres.fields.jsonb.JSONField(blank=True, null=True)),
                ('corpus', models.TextField(blank=True)),
                ('file', models.FileField(storage=django.core.files.storage.FileSystemStorage(location='/Users/andrei/.absa/models/w2v'), upload_to='')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('available', models.BooleanField(default=False)),
            ],
            options={
                'verbose_name': 'Word2Vec',
                'verbose_name_plural': 'Word2Vec models',
            },
        ),
    ]
