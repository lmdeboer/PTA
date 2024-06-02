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


def semantics(tokens, semantics_file, label, text):
    ambiguous_words = count_ambiguous_words(tokens)
    unique_synsets_count = unique_synsets(tokens)
    hypernyms = get_noun_hypernyms(tokens)
    most_common_hypernyms = common_hypernyms(hypernyms)
    num_nouns_without_synsets, nouns_without_synsets = count_tokens_without_synsets(text)
    num_named_entities = count_named_entities(text)
    num_unique_entities = count_unique_entities(text)
    num_clusters, avg_chain_len, max_chain_len = count_coreference(text)
    num_human_nouns_without_synsets, human_nouns_without_synsets = count_tokens_without_synsets(tokens)

    semantics_file.write(f"{label} articles\n")
    semantics_file.write(f"Number of ambiguous words: {ambiguous_words}\n")
    semantics_file.write(f"Number of unique synsets: {unique_synsets_count}\n")
    semantics_file.write(f"10 most common hypernyms in {label} text: {most_common_hypernyms}\n")
    semantics_file.write(f"Amount of tokens in {label} articles that have no synsets: {num_nouns_without_synsets}\n")
    semantics_file.write(f"Top 10 tokens without synsets in {label} articles: {nouns_without_synsets}\n")
    semantics_file.write(f"Number of named entities: {num_named_entities}\n")
    semantics_file.write(f"Number of unique named entities: {num_unique_entities}\n")
    semantics_file.write(f"Number of coreference clusters: {num_clusters}\n")
    semantics_file.write(f"Average length of a coreference chain: {avg_chain_len}\n")
    semantics_file.write(f"Max length of a coreference chain: {max_chain_len}\n")
    semantics_file.write("----------------------------------------------------------\n")


def pragmatic(text, pf):
    sorted_word_frequency, polarity, subjectivity, evaluation = sentiment_analysis_tb(text)
    pf.write('Sentiment Analysis - SpacyTextBlob\n')
    pf.write("\n Polarity: " + str(polarity))
    pf.write("\n Subjectivity: " + str(subjectivity) + '\n\n')
    for frequency in sorted_word_frequency[:10]:
        pf.write("\nMost frequent high-sentiment words: " + frequency[0] + ": " + str(frequency[1]))
    if evaluation:
        pf.write("\n -> Evaluation: AI-generated text")
    else:
        pf.write("\n -> Evaluation: Human-generated text")

    high_sentiment_ratio, evaluation_pos_sents, evaluation_neg_sents, positive_sents_perc, negative_sents_perc, pos_neg_ratio = sentiment_ratios(text)
    pf.write("\n\nHigh Sentiment Sentences Percentage: " + str(high_sentiment_ratio))
    pf.write("\n\nPositive Sentences Percentage: " + str(positive_sents_perc))
    if evaluation_pos_sents:
        pf.write("\n -> Evaluation: AI-generated text")
    else:
        pf.write("\n -> Evaluation: Human-generated text")
    pf.write("\n\nNegative Sentences Percentage: " + str(negative_sents_perc))
    if evaluation_neg_sents:
        pf.write("\n -> Evaluation: AI-generated text")
    else:
        pf.write("\n -> Evaluation: Human-generated text\n")
    pf.write("\n\nPositive Negative Ratio: " + str(pos_neg_ratio))

    polarity, evaluation_as = sentiment_analysis_asent(text)
    pf.write('\n\nSentiment Analysis - Asent\n')
    pf.write('\nSentiment: ' + str(polarity))
    if evaluation_as:
        pf.write("\n -> Evaluation: AI-generated text")
    else:
        pf.write("\n -> Evaluation: Human-generated text")

    TraF = discourse_analysis(text)
    pf.write('\n\n Discourse Features\n')
    pf.write('\nReadability Features: ' + str(TraF))
    pf.write("\n----------------------------------------------------------\n")

    evaluations = [evaluation, evaluation_pos_sents, evaluation_neg_sents, evaluation_as]
    true_count = sum(evaluation == True for evaluation in evaluations)
    if true_count > 2:
        final_evaluation = True
        pf.write('\n\n Final Evaluation: AI-generated text')
    else:
        final_evaluation = False
        pf.write('\n\n Final Evaluation: Human-generated text')

    return final_evaluation

def final_evaluation(ev_syntax, ev_semantics, ev_pragmatics):
    evaluations = [ev_syntax, ev_semantics, ev_pragmatics]
    true_count = sum(evaluation == True for evaluation in evaluations)
    if true_count > 2:
        final_evaluation = True
    else:
        final_evaluation = False


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
        semantics(ai_tokens, semantics_file, "Artificial", ai_text)
        semantics(human_tokens, semantics_file, "Human", human_text)

    with open('pragmatics.txt', 'w', encoding='utf-8') as pf:
        pf.write('Human articles\n\n')
        pragmatic(human_text, pf)
        pf.write('\n\nAI articles\n\n')
        pragmatic(ai_text, pf)




if __name__ == '__main__':
    main()
