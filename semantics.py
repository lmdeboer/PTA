# Laura de Boer, Dertje Roggeveen, Roshana Vegter, Julian Paagman
# Group 2
# syntax.py


from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from collections import Counter


def unique_synsets(tokens):
    """
    Count the amount of unique synsets in the token list.
    :param tokens: List of tokens.
    :return: Number of unique synsets.
    """
    synsets_set = set()
    for token in tokens:
        # Get wordnet synsets for each token.
        synsets = wn.synsets(token)
        for synset in synsets:
            synsets_set.add(synset)
    return len(synsets_set)


def get_noun_hypernyms(tokens):
    """
    Get noun hypernyms for a given list of tokens.
    :param tokens: List of tokens.
    :return: List of hypernyms.
    """
    hypernyms = []
    for token in tokens:
        synsets = wn.synsets(token, pos=wn.NOUN)
        for synset in synsets:
            hypernyms.extend(synset.hypernyms())
    return hypernyms


def common_hypernyms(hypernyms):
    """
    Find the 10 most common hypernyms in the given list of tokens.
    :param tokens: List of tokens.
    :return: List of the 10 most common hypernyms.
    """
    hypernym_counter = Counter()
    for hypernym in hypernyms:
        hypernym_counter[hypernym.name()] += 1
    return hypernym_counter.most_common(10)


def count_ambiguous_words(tokens):
    """
    Count the number of ambiguous words in the list of tokens.
    :param tokens: List of tokens.
    :return: Number of ambiguous words.
    """
    num_ambiguous = 0
    for token in tokens:
        # Check token has more than 1 synsets
        if len(wn.synsets(token)) > 1:
            num_ambiguous += 1
    return num_ambiguous


def count_tokens_without_synsets(tokens):
    """
    Count the number of tokens that have no synsets in WordNet,
    and return the total count and the 10 most common tokens without synsets.
    :param tokens: List of tokens.
    :return: Total count of tokens without synsets and the 10 most common tokens without synsets.
    """
    num_tokens_without_synsets = 0
    tokens_without_synsets = []
    
    for token in tokens:
        # Get synsets of tokens.
        synsets = wn.synsets(token)
        # If it has no synsets add to count and add token to the list.
        if not synsets:
            num_tokens_without_synsets += 1
            tokens_without_synsets.append(token)
    
    # Count number of tokens without synsets
    total_count = num_tokens_without_synsets
    common_tokens = Counter(tokens_without_synsets).most_common(10)
    return total_count, common_tokens
