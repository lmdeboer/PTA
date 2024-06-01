# Laura de Boer, Dertje Roggeveen, Roshana Vegter, Julian Paagman
# Group 2
# nlp.py

from morphology import *
from syntax import *
from semantics import *
from pragmatics import *


def syntax(text, syntax_file):
    common_chunks = Counter(get_chunks(text))
    syntax_file.write(f"10 Most Common Noun Chunks:\n")
    for chunk, frequency in common_chunks.most_common(10):
        syntax_file.write(f"{chunk}: {frequency}\n")

    common_proper_noun_chunks = Counter(get_chunks(text, filter_root='PROPN'))
    syntax_file.write(f"\n10 Most Common Noun Chunks with Proper Noun Roots:\n")
    for chunk, frequency in common_proper_noun_chunks.most_common(10):
        syntax_file.write(f"{chunk}: {frequency}\n")

    tags = []
    for word in text:
        tags.append(word.pos_)

    most_frequent_tags, least_frequent_tags = count_tags(tags)
    syntax_file.write(f"10 Most frequent POS tags: {most_frequent_tags}\n")
    syntax_file.write(f"10 Least frequent POS tags: {least_frequent_tags}\n")

    syntax_file.write(f"Number of unique POS tags: {count_unique_tags(tags)}\n")

    syntax_file.write(f"Amount of stop words: {count_stop_words(text)}\n")
    syntax_file.write(f"Average sentence length: {average_length(text)}\n")
    syntax_file.write(f"Number of misspelled words: {count_misspelled_words(text)}\n")

    syntax_file.write(f"Most common specified dependency: {most_frequent_asked_dependency(text, 'ADJ')}\n")

    pos_avg_positions = pos_tags_distribution(text)
    syntax_file.write(f"Average position of POS tags in sentences: {pos_avg_positions}\n")

    syntax_file.write("----------------------------------------------------------\n")
    syntax_file.write("\n")


def semantics(tokens, semantics_file, label):
    ambiguous_words = count_ambiguous_words(tokens)
    unique_synsets_count = unique_synsets(tokens)
    hypernyms = get_noun_hypernyms(tokens)
    most_common_hypernyms = common_hypernyms(hypernyms)
    num_nouns_without_synsets, nouns_without_synsets = count_tokens_without_synsets(tokens)

    semantics_file.write(f"{label} articles\n")
    semantics_file.write(f"Number of ambiguous words: {ambiguous_words}\n")
    semantics_file.write(f"Number of unique synsets: {unique_synsets_count}\n")
    semantics_file.write(f"10 most common hypernyms in {label} text: {most_common_hypernyms}\n")
    semantics_file.write(f"Amount of tokens in {label} articles that have no synsets: {num_nouns_without_synsets}\n")
    semantics_file.write(f"Top 10 tokens without synsets in {label} articles: {nouns_without_synsets}\n")
    semantics_file.write("----------------------------------------------------------\n")


def pragmatic(text):
    sorted_h_word_frequency, sorted_ai_word_frequency, human_polarity, human_subjectivity, ai_polarity, ai_subjectivity = sentiment_analysis_tb(
    human_text, ai_text)
    pf.write('Sentiment Analysis - SpacyTextBlob\n')
    pf.write("\nHuman Polarity: " + str(human_polarity))
    pf.write("\nAI Polarity: " + str(ai_polarity))
    pf.write("\nHuman Subjectivity: " + str(human_subjectivity))
    pf.write("\nAI Subjectivity: " + str(ai_subjectivity))
    for frequency in sorted_h_word_frequency[:10]:
        pf.write("\nMost frequent high-sentiment words in human text: " + frequency[0] + ":" + str(frequency[1]))
    for frequency in sorted_ai_word_frequency[:10]:
        pf.write("\nMost frequent high-sentiment words in AI text: " + frequency[0] + ":" + str(frequency[1]))

    EnDF_AI, EnGF_AI, TraF_AI, EnDF_H, EnGF_H, TraF_H = discourse_analysis(human_text, ai_text)
    pf.write('\n Discourse Features\n')
    pf.write('\nHuman Entity Density Features: ' + str(EnDF_H))
    pf.write('\nAI Entity Density Features: ' + str(EnDF_AI))
    pf.write('\nHuman Entity Grid Features: ' + str(EnGF_H))
    pf.write('\nAI Entity Grid Features: ' + str(EnGF_AI))
    pf.write('\nHuman Readability Features: ' + str(TraF_H))
    pf.write('\nAI Readability Features: ' + str(TraF_AI))

    human_sentiment, ai_sentiment = sentiment_analysis_asent(human_text, ai_text)
    pf.write('\nSentiment Analysis - Asent\n')
    pf.write('Human sentiment: ' + str(human_sentiment))
    pf.write('\nAI sentiment: ' + str(ai_sentiment))


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

    with open('semantics.txt', 'w', encoding='utf-8') as semantics_file:
        semantics(ai_tokens, semantics_file, "Artificial")
        semantics(human_tokens, semantics_file, "Human")


    semantics(human_text)
    semantics(ai_text)

    with open('pragmatics.txt', 'w', encoding='utf-8') as pf: 
        pragmatic(human_text, ai_text, pf)


if __name__ == '__main__':
    main()
