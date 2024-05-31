import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
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


def main():
    nltk.download('wordnet')
    nlp = spacy.load("en_core_web_sm")
    
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
    num_human_nouns_without_synsets, human_nouns_without_synsets = count_tokens_without_synsets(human_tokens)
    
    # Print human results
    print("Human articles")
    print(f"Number of ambiguous words: {human_ambiguous_words}")
    print(f"Number of unique synsets: {human_unique_synsets}")
    print(f"10 most common hypernyms in human articles: {human_most_common_hypernyms}")
    print(f"Amount of tokens in human articles that have no synsets: {num_human_nouns_without_synsets}")
    print(f"Top 10 tokens without synsets in human articles: {human_nouns_without_synsets}")
    print("----------------------------------------------------------")
    
    # Process AI articles
    ai_text = read_file(ai_file)
    ai_lemmas = lemmatization(ai_text)
    ai_tokens = tokenization(ai_text)
    
    # Analyze AI articles
    ai_ambiguous_words = count_ambiguous_words(ai_tokens)
    ai_unique_synsets = unique_synsets(ai_tokens)
    ai_hypernyms = get_noun_hypernyms(ai_tokens)
    ai_most_common_hypernyms = common_hypernyms(ai_hypernyms)
    num_ai_nouns_without_synsets, ai_nouns_without_synsets = count_tokens_without_synsets(ai_tokens)

    # Print AI articles results
    print("AI generated articles")
    print(f"Number of ambiguous words: {ai_ambiguous_words}")
    print(f"Number of unique synsets: {ai_unique_synsets}")
    print(f"10 most common hypernyms in AI text: {ai_most_common_hypernyms}")
    print(f"Amount of tokens in AI articles that have no synsets: {num_ai_nouns_without_synsets}")
    print(f"Top 10 tokens without synsets in AI articles: {ai_nouns_without_synsets}")
    print("----------------------------------------------------------")

if __name__ == '__main__':
    main()
