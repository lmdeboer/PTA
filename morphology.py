# Laura de Boer, Dertje Roggeveen, Roshana Vegter
# Group 2
# morphology.py

import json
import spacy
from collections import Counter


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


def get_chunks(doc, filter_root=None):
    """Get noun chunks from a spaCy document, optionally filtered by
    root token properties
    :param doc:
    :param filter_root:
    :return: """
    if filter_root:
        chunks = (chunk.text for chunk in doc.noun_chunks if
                  chunk.root.pos_ == filter_root)
    else:
        chunks = (chunk.text for chunk in doc.noun_chunks)
    return chunks


def main():
    human_file = 'human.jsonl'
    ai_file = 'group2.jsonl'
    human_text = read_file(human_file)
    ai_text = read_file(ai_file)
    human_tokens = tokenization(human_text)
    ai_tokens = tokenization(ai_text)
    ai_lemmas = lemmatization(ai_text)
    human_lemmas = lemmatization(human_text)

    common_chunks = Counter(get_chunks(ai_text))
    print("10 Most Common Noun Chunks in ai text:")
    for chunk, frequency in common_chunks.most_common(10):
        print(f"{chunk}: {frequency}")

    common_chunks = Counter(get_chunks(human_text))
    print("10 Most Common Noun Chunks in human text:")
    for chunk, frequency in common_chunks.most_common(10):
        print(f"{chunk}: {frequency}")


if __name__ == '__main__':
    main()