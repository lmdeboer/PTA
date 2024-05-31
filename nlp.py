import json
import spacy
from syntax import *


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


def syntax(text_ai, syntax_file):
    common_chunks_ai = Counter(get_chunks(text_ai))
    syntax_file.write("10 Most Common Noun Chunks in text:\n")
    for chunk, frequency in common_chunks_ai.most_common(10):
        syntax_file.write(f"{chunk}: {frequency}\n")

    common_proper_noun_chunks_ai = Counter(get_chunks(text_ai, filter_root='PROPN'))
    syntax_file.write("\n10 Most Common Noun Chunks with Proper Noun Roots in text:\n")
    for chunk, frequency in common_proper_noun_chunks_ai.most_common(10):
        syntax_file.write(f"{chunk}: {frequency}\n")

    tags_ai = []
    for word in text_ai:
        tags_ai.append(word.pos_)
    syntax_file.write("10 Most frequent POS tags in text: " +
                      str(count_tags(tags_ai)) + "\n")

    syntax_file.write("Most common specified dependency in text: " +
                      str(most_frequent_asked_dependency(text_ai, 'ADJ')) + "\n")

    syntax_file.write("----------------------------------------------------------\n")
    syntax_file.write("\n")

def semantics(text):
    pass


def pragmatic(text):
    pass


def main():
    human_file = 'human.jsonl'
    ai_file = 'group2.jsonl'
    human_text = read_file(human_file)
    ai_text = read_file(ai_file)
    human_tokens = tokenization(human_text)
    ai_tokens = tokenization(ai_text)
    ai_lemmas = lemmatization(ai_text)
    human_lemmas = lemmatization(human_text)

    with open('syntax.txt', 'w', encoding='utf-8') as syntax_file:
        syntax_file.write("Artificial articles\n")
        syntax(ai_text, syntax_file)
        syntax_file.write("Human articles\n")
        syntax(human_text, syntax_file)

    semantics(human_text)
    semantics(ai_text)

    pragmatic(human_text)
    pragmatic(ai_text)


if __name__ == '__main__':
    main()
