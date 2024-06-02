# Laura de Boer, Dertje Roggeveen, Roshana Vegter, Julian Paagman
# Group 2
# nlp.py


from morphology import *
from syntax import *
from semantics import *
from pragmatics import *


# Author: Dertje Roggeveen
def syntax(text, syntax_file):
    common_chunks = Counter(get_chunks(text))
    syntax_file.write(f"10 Most Common Noun Chunks:\n")
    for chunk, frequency in common_chunks.most_common(10):
        syntax_file.write(f"{chunk}: {frequency}\n")

    common_proper_noun_chunks = Counter(get_chunks(text, filter_root='PROPN'))
    syntax_file.write(f"\n10 Most Common Noun Chunks with Proper "
                      f"Noun Roots:\n")
    for chunk, frequency in common_proper_noun_chunks.most_common(10):
        syntax_file.write(f"{chunk}: {frequency}\n")

    tags = []
    for word in text:
        tags.append(word.pos_)

    most_frequent_tags, least_frequent_tags = count_tags(tags)
    syntax_file.write(f"10 Most frequent POS tags: {most_frequent_tags}\n")
    syntax_file.write(f"10 Least frequent POS tags: {least_frequent_tags}\n")

    syntax_file.write(f"Number of unique POS tags: "
                      f"{count_unique_tags(tags)}\n")

    stop_words, stop_words_eval = count_stop_words(text)
    syntax_file.write(f"Amount of stop words: {stop_words}\n")
    if stop_words_eval:
        syntax_file.write("\n -> Evaluation: Human-generated text")
    else:
        syntax_file.write("\n -> Evaluation: AI-generated text")

    sent_length, length_eval = average_length(text)
    syntax_file.write(f"Average sentence length: {sent_length}\n")
    if length_eval:
        syntax_file.write("\n -> Evaluation: Human-generated text")
    else:
        syntax_file.write("\n -> Evaluation: AI-generated text")

    misspelled_words, misspelled_eval = count_misspelled_words(text)
    syntax_file.write(f"Number of misspelled words: {misspelled_words}\n")
    if misspelled_eval:
        syntax_file.write("\n -> Evaluation: Human-generated text")
    else:
        syntax_file.write("\n -> Evaluation: AI-generated text")

    syntax_file.write(f"Most common specified dependency: "
                      f"{most_frequent_asked_dependency(text, 'ADJ')}\n")

    pos_avg_positions = pos_tags_distribution(text)
    syntax_file.write(f"Average position of POS tags in sentences: "
                      f"{pos_avg_positions}\n")

    syntax_file.write("-----------------------------------------"
                      "-----------------\n")
    syntax_file.write("\n")

    evaluations = [stop_words_eval, length_eval, misspelled_eval]
    true_count = sum(evaluation is True for evaluation in evaluations)
    if true_count >= 2:
        final_evaluation = True
    else:
        final_evaluation = False

    return final_evaluation


def semantics(tokens, semantics_file, label, text):
    # Author Roshana Vegter
    semantics_file.write(f"{label} articles\n")
    ambiguous_words = count_ambiguous_words(tokens)
    semantics_file.write(f"Number of ambiguous words: {ambiguous_words}\n")
    average_amount_ambiguous_words, ambig_words_eval = \
        average_ambiguous_words(tokens)
    semantics_file.write(f"Average amount of ambiguous words: "
                         f"{average_amount_ambiguous_words}\n")
    if ambig_words_eval:
        semantics_file.write("-> Evaluation: Human-generated text\n")
    else:
        semantics_file.write("-> Evaluation: AI-generated text\n")

    num_tokens_without_synsets, tokens_without_synsets, synset_eval = \
        count_tokens_without_synsets(tokens)
    semantics_file.write(f"Amount of tokens in {label} articles that "
                         f"have no synsets: {num_tokens_without_synsets}\n")
    semantics_file.write(f"Top 10 tokens without synsets in {label} "
                         f"articles: {tokens_without_synsets}\n")
    if synset_eval:
        semantics_file.write("-> Evaluation: Human-generated text\n")
    else:
        semantics_file.write("-> Evaluation: AI-generated text\n")

    unique_synsets_count = unique_synsets(tokens)
    semantics_file.write(f"Number of unique synsets: {unique_synsets_count}\n")

    avg_token_without_synset, avg_token_without_synset_eval = \
        average_tokens_without_synsets(tokens, num_tokens_without_synsets)
    semantics_file.write(f"Average amount of tokens without synsets: "
                         f"{avg_token_without_synset}")

    avg_token_without_synset, avg_token_without_synset_eval = average_tokens_without_synsets(tokens, num_tokens_without_synsets)
    semantics_file.write(f"Average amount of tokens without synsets: {avg_token_without_synset}\n")

    if avg_token_without_synset_eval:
        semantics_file.write("-> Evaluation: Human-generated text\n")
    else:
        semantics_file.write("-> Evaluation: AI-generated text\n")

    hypernyms = get_noun_hypernyms(tokens)
    most_common_hypernyms, hypernym_eval = common_hypernyms(hypernyms)
    semantics_file.write(f"10 most common hypernyms in {label} text: "
                         f"{most_common_hypernyms}\n")
    if hypernym_eval:
        semantics_file.write("-> Evaluation: Human-generated text\n")
    else:
        semantics_file.write("-> Evaluation: AI-generated text\n")
    # End part Roshana Vegter

    # Author Julian Paagman
    num_named_entities, named_entities_eval = count_named_entities(text)
    num_unique_entities, unique_entities_eval = count_unique_entities(text)
    num_clusters, avg_chain_len, max_chain_len = count_coreference(text)

    semantics_file.write(f"Number of named entities: {num_named_entities}\n")
    if named_entities_eval:
        semantics_file.write(f"-> Evaluation: Human-generated text\n")
    else:
        semantics_file.write(f"-> Evaluation: AI-generated text\n")
    semantics_file.write(f"Number of unique named entities: "
                         f"{num_unique_entities}\n")
    if unique_entities_eval:
        semantics_file.write(f"-> Evaluation: Human-generated text\n")
    else:
        semantics_file.write(f"-> Evaluation: AI-generated text\n")
    semantics_file.write(f"Number of coreference clusters: "
                         f"{num_clusters}\n")
    semantics_file.write(f"Average length of a coreference chain: "
                         f"{avg_chain_len}\n")
    semantics_file.write(f"Max length of a coreference chain: "
                         f"{max_chain_len}\n")
    semantics_file.write("------------------------------------"
                         "----------------------\n")

    evaluations = [ambig_words_eval, avg_token_without_synset_eval,
                   hypernym_eval, synset_eval, named_entities_eval,
                   unique_entities_eval]
    true_count = sum(evaluation is True for evaluation in evaluations)
    if true_count > 3:
        final_evaluation = True
        semantics_file.write('\n\n Final Evaluation: AI-generated text')
    else:
        final_evaluation = False
        semantics_file.write('\n\n Final Evaluation: Human-generated text')

    return final_evaluation


def pragmatic(text, pf):
    sorted_word_frequency, polarity, subjectivity, evaluation = \
        sentiment_analysis_tb(text)
    pf.write('Sentiment Analysis - SpacyTextBlob\n')
    pf.write("\n Polarity: " + str(polarity))
    pf.write("\n Subjectivity: " + str(subjectivity) + '\n\n')
    for frequency in sorted_word_frequency[:10]:
        pf.write("\nMost frequent high-sentiment words: " +
                 frequency[0] + ": " + str(frequency[1]))
    if evaluation:
        pf.write("\n -> Evaluation: AI-generated text")
    else:
        pf.write("\n -> Evaluation: Human-generated text")

    high_sentiment_ratio, evaluation_pos_sents, \
        evaluation_neg_sents, positive_sents_perc, \
        negative_sents_perc, pos_neg_ratio = sentiment_ratios(text)

    pf.write("\n\nHigh Sentiment Sentences Percentage: " +
             str(high_sentiment_ratio))
    pf.write("\n\nPositive Sentences Percentage: " +
             str(positive_sents_perc))
    if evaluation_pos_sents:
        pf.write("\n -> Evaluation: AI-generated text")
    else:
        pf.write("\n -> Evaluation: Human-generated text")
    pf.write("\n\nNegative Sentences Percentage: " +
             str(negative_sents_perc))
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

    evaluations = [evaluation, evaluation_pos_sents,
                   evaluation_neg_sents, evaluation_as]
    true_count = sum(evaluation is True for evaluation in evaluations)
    if true_count > 2:
        final_evaluation = True
        pf.write('\n\n Final Evaluation: AI-generated text')
    else:
        final_evaluation = False
        pf.write('\n\n Final Evaluation: Human-generated text')

    pf.write("\n-----------------------------------------"
             "-----------------\n")

    return final_evaluation

# author: Laura de Boer
def final_evaluation(ev_syntax, ev_semantics, ev_pragmatics, ef):
    evaluations = [ev_syntax, ev_semantics, ev_pragmatics]
    true_count = sum(evaluation is True for evaluation in evaluations)
    if true_count > 1:
        final_evaluation = True
        ef.write('\n\n Final Evaluation: AI-generated text')
    else:
        final_evaluation = False
        ef.write('\n\n Final Evaluation: Human-generated text')
    return final_evaluation


def main():
    file = 'human.jsonl'
    texts = read_file(file)

    for doc in texts.keys():
        tokens = tokenization(doc)
        lemmas = lemmatization(doc)

        with open('syntax.txt', 'a', encoding='utf-8') as syntax_file:
            syntax_file.write("Articles\n")
            ev_syntax = syntax(doc, syntax_file)

        with open('semantics.txt', 'a', encoding='utf-8') as semantics_file:
            ev_semantics = semantics(tokens, semantics_file, "Articles", doc)

        with open('pragmatics.txt', 'a', encoding='utf-8') as pf:
            pf.write('Articles\n\n')
            ev_pragmatics = pragmatic(doc, pf)

        with open('evaluation.txt', 'a', encoding='utf-8') as ef:
            final_evaluation(ev_syntax, ev_semantics, ev_pragmatics, ef)

        if final_evaluation:
            texts[doc] == "AI"
        else:
            texts[doc] == "Human"
    print(texts)


if __name__ == '__main__':
    main()
