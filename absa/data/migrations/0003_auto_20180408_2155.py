# Generated by Django 2.0.4 on 2018-04-08 21:55

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0002_auto_20180408_0203'),
    ]

    operations = [
        migrations.AlterModelTable(
            name='sentencebatch',
            table='data_sentence_batch',
        ),
    ]
