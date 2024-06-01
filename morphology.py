# Laura de Boer, Dertje Roggeveen, Roshana Vegter, Julian Paagman
# Group 2
# morphology.py


import json
import spacy


def read_file(file_name):
    """
    Function which reads the json file, and appends each article to the list texts, as well as apply a spacy object
    on the document.
    :param file_name: Json file to be read
    :return: text as a spacy object.
    """
    texts = []
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        # half of the files, could not process more due to character limit
        for line in lines[:326]:
            json_obj = json.loads(line.strip())
            text = json_obj.get('text', '')
            texts.append(text)

    all_text = ' '.join(texts)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(all_text)

    return doc


def stemming(text):
    pass


def lemmatization(text):
    """
    Function which lemmatizes the text.
    :param text: Text to be lemmatized.
    :return: Lemmatized text.
    """

    return [token.lemma_ for token in text]


def tokenization(text):
    """
    Function which tokenizes the text.
    :param text: Text to be tokenized.
    :return: Tokenized text.
    """

    return [token.text.strip() for token in text]