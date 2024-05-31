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


def count_tags(tags):
    """Count the frequency of each POS tag and return the
    10 most frequent tags
    :param tags:
    :return: most frequent POS tags.
    """
    unique_pos = len(Counter(tags))
    print("Number of unique POS tags: " + str(unique_pos))

    pos = {}
    # Loop through tags
    for tag in tags:
        if tag in pos:
            pos[tag] += 1
        else:
            pos[tag] = 1

    # count 10 most frequent values
    N = 10
    most_frequent = dict(sorted(pos.items(),
                                key=lambda x: x[1], reverse=True)[:N])

    return most_frequent


def most_frequent_asked_dependency(text, dependency):
    """Find the most frequent adjective in the given text"""
    dependencies = Counter()
    for token in text:
        if token.pos_ == dependency:
            dependencies[token.text] += 1

    most_common_dependency = dependencies.most_common(1)

    return dependency, most_common_dependency


def get_chunks(doc, filter_root=None):
    """Get noun chunks from a spaCy document, optionally filtered by
    root token properties
    :param doc
    :param filter_root
    :return: noun chunks
    """
    if filter_root:
        chunks = (chunk.text for chunk in doc.noun_chunks if
                  chunk.root.pos_ == filter_root)
    else:
        chunks = (chunk.text for chunk in doc.noun_chunks)

    return chunks


def extract_subject_pnoun_chunks(doc):
    """Extract noun chunks where the root token is a proper noun and
    the subject of the sentence.
    :param doc
    :return: proper and subject root noun chunks
    """
    subject_pnoun_chunks = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == 'nsubj' and token.pos_ == 'PROPN':
                root_token = token
                break
        else:
            continue
        for chunk in sent.noun_chunks:
            if chunk.root == root_token:
                subject_pnoun_chunks.append(chunk.text)

    return subject_pnoun_chunks


def main():
    human_file = 'human.jsonl'
    ai_file = 'group2.jsonl'
    human_text = read_file(human_file)
    ai_text = read_file(ai_file)
    human_tokens = tokenization(human_text)
    ai_tokens = tokenization(ai_text)
    ai_lemmas = lemmatization(ai_text)
    human_lemmas = lemmatization(human_text)

    common_chunks_ai = Counter(get_chunks(ai_text))
    print("10 Most Common Noun Chunks in ai text:")
    for chunk, frequency in common_chunks_ai.most_common(10):
        print(f"{chunk}: {frequency}")

    common_chunks_human = Counter(get_chunks(human_text))
    print("10 Most Common Noun Chunks in human text:")
    for chunk, frequency in common_chunks_human.most_common(10):
        print(f"{chunk}: {frequency}")

    common_proper_noun_chunks_ai = Counter(get_chunks(ai_text, filter_root='PROPN'))
    print("\n10 Most Common Noun Chunks with Proper Noun Roots in ai text:")
    for chunk, frequency in common_proper_noun_chunks_ai.most_common(10):
        print(f"{chunk}: {frequency}")

    common_proper_noun_chunks_human = Counter(get_chunks(human_text, filter_root='PROPN'))
    print("\n10 Most Common Noun Chunks with Proper Noun Roots in ai text:")
    for chunk, frequency in common_proper_noun_chunks_human.most_common(10):
        print(f"{chunk}: {frequency}")

    tags_ai = []
    for word in ai_text:
        tags_ai.append(word.pos_)
    print("10 Most frequent POS tags in ai text: " +
          str(count_tags(tags_ai)))

    tags_human = []
    for word in human_text:
        tags_human.append(word.pos_)
    print("10 Most frequent POS tags in human text: " +
          str(count_tags(tags_human)))

    print("Most common specified dependency in ai text: " +
          str(most_frequent_asked_dependency(ai_text, 'ADJ')))

    print("Most common specified dependency in human text: " +
          str(most_frequent_asked_dependency(human_text, 'ADJ')))


if __name__ == '__main__':
    main()