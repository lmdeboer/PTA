# Laura de Boer, Dertje Roggeveen, Roshana Vegter, Julian Paagman
# Group 2
# syntax.py


from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from fastcoref import spacy_component
import spacy
import json
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
    nlp.add_pipe("fastcoref")
    doc = nlp(all_text)
    return doc


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


def count_named_entities(tokens):
    """
    Count the number of named entities per tag
    param: List of tokens.
    returns: a dictionary containing every named entity tag and its frequency
    """
    count = Counter()
    for ent in tokens.ents:
        count[ent.label_] += 1
    return count


def count_unique_entities(tokens):
    """
    Counts the number of unique entities
    param: List of tokens.
    returns: The number of unique named entity tags
    """
    unique = []
    for ent in tokens.ents:
        if ent not in unique:
            unique.append(ent)
    return len(unique)


def count_coreference(doc):
    """
    Counts the number of coreference clusters, average length of a cluster and maximum length of a cluster in the text
    param: A text.
    returns: The number, average length and maximum length of clusters in the text
    """
    clusters = doc._.coref_clusters

    num_clust = len(clusters)
    total_chain_len = 0
    max_chain_len = 0

    for cluster in clusters:
        chain_len = len(cluster)
        total_chain_len += chain_len
        max_chain_len = max(max_chain_len, chain_len)

    avg_chain_len= total_chain_len / num_clust if num_clust > 0 else 0

    return num_clust, avg_chain_len, max_chain_len


def main():
    nltk.download('wordnet')
    
    # Files
    human_file = 'human.jsonl'
    ai_file = 'group2.jsonl'
    
    # Process human articles
    human_text = read_file(human_file)
    human_lemmas = lemmatization(human_text)
    human_tokens = tokenization(human_text)
    
    # Analyze human articles
    human_ambiguous_words = count_ambiguous_words(human_tokens)
    human_unique_synsets = unique_synsets(human_tokens)
    human_hypernyms = get_noun_hypernyms(human_tokens)
    human_most_common_hypernyms = common_hypernyms(human_hypernyms)
    human_num_named_entities = count_named_entities(human_tokens)
    human_num_unique_entities = count_unique_entities(human_tokens)
    human_num_clusters, human_avg_chain_len, human_max_chain_len = count_coreference(human_tokens)
    num_human_nouns_without_synsets, human_nouns_without_synsets = count_tokens_without_synsets(human_tokens)
    
    # Process AI articles
    ai_text = read_file(ai_file)
    ai_lemmas = lemmatization(ai_text)
    ai_tokens = tokenization(ai_text)
    
    # Analyze AI articles
    ai_ambiguous_words = count_ambiguous_words(ai_tokens)
    ai_unique_synsets = unique_synsets(ai_tokens)
    ai_hypernyms = get_noun_hypernyms(ai_tokens)
    ai_most_common_hypernyms = common_hypernyms(ai_hypernyms)
    ai_num_named_entities = count_named_entities(ai_tokens)
    ai_num_unique_entities = count_unique_entities(ai_tokens)
    ai_num_clusters, ai_avg_chain_len, ai_max_chain_len = count_coreference(ai_tokens)
    num_ai_nouns_without_synsets, ai_nouns_without_synsets = count_tokens_without_synsets(ai_tokens)


    with open('semantics.txt', 'w', encoding='utf-8') as semantics_file:
        # Wite human article results to file
        semantics_file.write("Human articles\n")
        semantics_file.write(f"Number of ambiguous words: {human_ambiguous_words}\n")
        semantics_file.write(f"Number of unique synsets: {human_unique_synsets}\n")
        semantics_file.write(f"10 most common hypernyms in human articles: {human_most_common_hypernyms}\n")
        semantics_file.write(f"Amount of tokens in human articles that have no synsets: {num_human_nouns_without_synsets}\n")
        semantics_file.write(f"Top 10 tokens without synsets in human articles: {human_nouns_without_synsets}\n")
        semantics_file.write(f"Number of named entities: {human_num_named_entities}\n")
        semantics_file.write(f"Number of unique named entities: {human_num_unique_entities}\n")
        semantics_file.write(f"Number of coreference clusters: {human_num_clusters}\n")
        semantics_file.write(f"Average length of a coreference chain: {human_avg_chain_len}\n")
        semantics_file.write(f"Max length of a coreference chain: {human_max_chain_len}\n")
        semantics_file.write("----------------------------------------------------------\n")
        
        # Write AI articles results to file.
        semantics_file.write("AI generated articles\n")
        semantics_file.write(f"Number of ambiguous words: {ai_ambiguous_words}\n")
        semantics_file.write(f"Number of unique synsets: {ai_unique_synsets}\n")
        semantics_file.write(f"10 most common hypernyms in AI text: {ai_most_common_hypernyms}\n")
        semantics_file.write(f"Amount of tokens in AI articles that have no synsets: {num_ai_nouns_without_synsets}\n")
        semantics_file.write(f"Top 10 tokens without synsets in AI articles: {ai_nouns_without_synsets}\n")
        semantics_file.write(f"Number of named entities: {ai_num_named_entities}\n")
        semantics_file.write(f"Number of unique named entities: {ai_num_unique_entities}\n")
        semantics_file.write(f"Number of coreference clusters: {ai_num_clusters}\n")
        semantics_file.write(f"Average length of a coreference chain: {ai_avg_chain_len}\n")
        semantics_file.write(f"Max length of a coreference chain: {ai_max_chain_len}\n")
        semantics_file.write("----------------------------------------------------------\n")

if __name__ == '__main__':
    main()
