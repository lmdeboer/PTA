# Laura de Boer, Dertje Roggeveen, Roshana Vegter, Julian Paagman
# Group 2
# syntax.py


from collections import Counter


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
