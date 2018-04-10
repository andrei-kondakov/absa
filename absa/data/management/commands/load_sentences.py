import os
import sys
import xml.etree.ElementTree as ET

from django.core.management.base import BaseCommand, CommandError

from data.models import TestSentence, TrainSentence


class Command(BaseCommand):
    help = 'Load sentences from file in the semeval format'

    def add_arguments(self, parser):
        parser.add_argument('filepath', type=str)

    def handle(self, *args, **options):
        filepath = options['filepath']
        filename, _ = os.path.splitext(os.path.basename(filepath))

        is_train_sentences = 'train' in filename.lower()
        is_test_setnences = 'test' in filename.lower()

        if not is_train_sentences and not is_test_setnences:
            raise CommandError(f'It is not possible to determine the type of sentences by file name')

        Sentence = TrainSentence if is_train_sentences else TestSentence

        with open(options['filepath'], encoding='utf8') as file:
            xml_content = file.read()
            xml_tree = ET.XML(xml_content)

            for sentence_elem in xml_tree.iter('sentence'):
                sid = sentence_elem.attrib.get('id')
                text = sentence_elem.find('text').text
                out_of_scope = 'OutOfScope' in sentence_elem.attrib
                categories = []
                polarities = []

                opinions = sentence_elem.find('Opinions')

                if opinions:
                    for opinion_elem in opinions.iter('Opinion'):
                        # target = opinion_elem.attrib.get('target', '')
                        category = opinion_elem.attrib.get('category')
                        polarity = opinion_elem.attrib.get('polarity')

                        if category not in categories:
                            categories.append(category)

                        if polarity not in polarities:
                            polarities.append(polarity)

                Sentence.objects.create(
                    sid=sid,
                    text=text,
                    out_of_scope=out_of_scope,
                    categories=sorted(categories),
                    polarities=sorted(polarities)
                )

        if not 'test' in sys.argv:
            self.stdout.write(self.style.SUCCESS(f'Loaded {Sentence.objects.count()} sentences from {filepath}'))
