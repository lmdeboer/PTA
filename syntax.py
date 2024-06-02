# Laura de Boer, Dertje Roggeveen, Roshana Vegter, Julian Paagman
# Group 2
# syntax.py


from collections import Counter
from spellchecker import SpellChecker



def count_tags(tags):
    """Count the frequency of each POS tag and return the
    10 most frequent tags
    :param tags:
    :return: most frequent POS tags.
    """
    pos = {}
    # Loop through tags
    for tag in tags:
        if tag in pos:
            pos[tag] += 1
        else:
            pos[tag] = 1

    # count 10 most frequent and least frequent values
    most_frequent = dict(sorted(pos.items(),
                                key=lambda x: x[1], reverse=True)[:10])

    # 10 least frequent tags (sorted in ascending order of frequency)
    least_frequent = dict(sorted(pos.items(), key=lambda x: x[1])[:10])

    return most_frequent, least_frequent


def count_unique_tags(tags):
    unique_pos = len(Counter(tags))

    return unique_pos


def count_stop_words(text):
    stop_words = 0
    words = [token.text for token in text if token.is_alpha]
    for word in text:
        if word.is_stop:
            stop_words += 1

    percentage_stop_words = stop_words / len(words) * 100

    evaluation = False
    if percentage_stop_words > 40:
        evaluation = True

    return stop_words, evaluation


def count_misspelled_words(text):
    spell = SpellChecker()
    words = [token.text for token in text if token.is_alpha]
    misspelled = spell.unknown(words)
    percentage_wrong = (len(misspelled) / len(words)) * 100

    evaluation = False
    if percentage_wrong > 15:
        evaluation = True

    return len(misspelled), evaluation

def average_length(text):
    """Calculates the average length of sentences in a given SpaCy `Doc`."""
    sent_length = 0
    for sentence in text:
        sent_length += len(sentence)

    evaluation = False
    if sent_length < 4.5:
        evaluation = True

    return round(sent_length / len(text), 2), evaluation


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


def pos_tags_distribution(doc):
    """
    Calculate where in a sentence each POS tag appears on average.
    :param doc: A spaCy Doc object containing the text.
    :return: A dictionary where keys are POS tags and values are the average positions in sentences.
    """
    # Dictionary to store the sum of positions and counts for each POS tag
    pos_positions = {}

    # Iterate through sentences in the doc
    for sent in doc.sents:
        # Iterate through tokens in the sentence
        for idx, token in enumerate(sent):
            pos_tag = token.pos_
            if pos_tag not in pos_positions:
                pos_positions[pos_tag] = {'sum_pos': 0, 'count': 0}

            pos_positions[pos_tag]['sum_pos'] += idx
            pos_positions[pos_tag]['count'] += 1

    # Calculate the average position for each POS tag
    pos_avg_positions = {tag: round(data['sum_pos'] / data['count'], 2) for tag, data in pos_positions.items()}

    return pos_avg_positions
