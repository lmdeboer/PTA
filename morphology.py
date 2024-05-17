# Laura de Boer, Dertje Roggeveen, Roshana Vegter
# Group 2
# morphology.py

import json


def read_file(file_name):
    texts = []
    with open('file_name', 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            text = json_obj.get('text', '')
            texts.append(text)

    for text in texts:
        return text

def stemming():
    pass


def lemmatization():
    pass


def tokenization():
    pass


def stop_words():
    pass


def spelling_correction():
    pass


def main():
    human_file = 'human.jsonl'
    ai_file = 'group2.jsonl'


if __name__ == '__main__':
    main()