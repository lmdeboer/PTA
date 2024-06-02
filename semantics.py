# Laura de Boer, Dertje Roggeveen, Roshana Vegter, Julian Paagman
# Group 2
# semantics.py


from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from fastcoref import spacy_component
from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("fastcoref")

# Author Roshana Vegter
def unique_synsets(tokens):
    """
    This functions counts the amount of unique synsets in a list of tokens.
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


# Author Roshana Vegter
def get_noun_hypernyms(tokens):
    """
    This function gets the hypernyms out of a list of tokens.
    :param tokens: List of tokens.
    :return: List of hypernyms.
    """
    hypernyms = []
    for token in tokens:
        synsets = wn.synsets(token, pos=wn.NOUN)
        for synset in synsets:
            hypernyms.extend(synset.hypernyms())
    return hypernyms


# Author Roshana Vegter
def common_hypernyms(hypernyms):
    """
    This function finds the 10 most common hypernyms in a list of tokens.
    :param tokens: List of tokens.
    :return: List of the 10 most common hypernyms.
    """
    hypernym_counter = Counter()
    for hypernym in hypernyms:
        hypernym_counter[hypernym.name()] += 1
    
    evaluation = False
    if hypernym_counter.most_common(2)[1][0] == 'time_period.n.01':
        evaluation = True

    return hypernym_counter.most_common(10), evaluation


# Author Roshana Vegter
def count_ambiguous_words(tokens):
    """
    This function counts the amount of ambiguous words in a list of tokens.
    :param tokens: List of tokens.
    :return: Number of ambiguous words.
    """
    num_ambiguous = 0
    for token in tokens:
        # Check token has more than 1 synsets
        if len(wn.synsets(token)) > 1:
            num_ambiguous += 1
    return num_ambiguous


# Author Roshana Vegter
def average_ambiguous_words(tokens):
    """
    This function calculates the average number of ambiguous words in a list of tokens.
    :param tokens: List of tokens.
    :return: Average number of ambiguous words.
    """
    total_ambiguous_words = count_ambiguous_words(tokens)
    total_tokens = len(tokens)
    average_ambiguous = total_ambiguous_words / total_tokens if total_tokens > 0 else 0

    evaluation = False
    if average_ambiguous > 0.52:
        evaluation = True

    return average_ambiguous, evaluation


# Author Roshana Vegter
def count_tokens_without_synsets(tokens):
    """
    This function counts the amount of tokens that have no synsets in WordNet,
    and returns te total count and the 10 most common words without synsets.
    :param tokens: List of tokens.
    :return: Amount of tokens without synsets and the 10 most common tokens
    without synsets.
    """
    num_tokens_without_synsets = 0
    tokens_without_synsets = []

    for token in tokens:
        if isinstance(token, str):
        # Get synsets of tokens.
            synsets = wn.synsets(token)
            # If it has no synsets add to count and add token to the list.
            if not synsets:
                num_tokens_without_synsets += 1
                tokens_without_synsets.append(token)
    
    # Count number of tokens without synsets
    total_count = num_tokens_without_synsets
    common_tokens = Counter(tokens_without_synsets).most_common(10)

    evaluation = False
    if common_tokens and common_tokens[0][0] == 'the':
        evaluation = True

    return total_count, common_tokens, evaluation


# Roshana Vegter
def average_tokens_without_synsets(tokens, tokens_without_synsets):
    """
    This function calculates the average number of tokens without synsets 
    in a list of tokens.
    :param tokens: List of lists of tokens.
    :return: Average number of tokens without synsets.
    """
    num_tokens_without_synsets = tokens_without_synsets

    avg_tokens_without_synsets = num_tokens_without_synsets / len(tokens)

    evaluation = False
    if avg_tokens_without_synsets > 0.4 :
        evaluation = True

    return avg_tokens_without_synsets, evaluation


# Author Julian Paagman
def count_named_entities(text):
    """
    Count the number of named entities per tag
    param: List of tokens.
    returns: a dictionary containing every named entity tag and its frequency
    """
    count = Counter()
    for ent in text.ents:
        count[ent.label_] += 1
    return count



# Author Julian Paagman
def count_unique_entities(text):
    """
    Counts the number of unique entities
    param: List of tokens.
    returns: The number of unique named entity tags
    """
    unique = []
    for ent in text.ents:
        if ent not in unique:
            unique.append(ent)
    return len(unique)


# Author Julian Paagman
def count_coreference(doc):
    """
    Counts the number of coreference clusters, average length of a cluster
    and maximum length of a cluster in the text
    param: A text.
    returns: The number, average length and maximum length of clusters
    in the text
    """
    clusters = doc._.coref_clusters

    if clusters is None:
        return 0, 0, 0

    num_clust = len(clusters)
    total_chain_len = 0
    max_chain_len = 0

    for cluster in clusters:
        chain_len = len(cluster)
        total_chain_len += chain_len
        max_chain_len = max(max_chain_len, chain_len)

    avg_chain_len = total_chain_len / num_clust if num_clust > 0 else 0

    return num_clust, avg_chain_len, max_chain_len
    
