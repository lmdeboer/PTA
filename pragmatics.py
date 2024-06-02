# Laura de Boer, Dertje Roggeveen, Roshana Vegter, Julian Paagman
# Group 2
# pragmatics.py

from collections import defaultdict
from spacytextblob.spacytextblob import SpacyTextBlob
from lingfeat import extractor
import asent
import spacy


def sentiment_analysis_tb(text):
    """
    Sentiment analysis using TextBlob. It first establishes the necessary objects for spacy. Moreover, the 10 most
    common high-sentiment words per text are found by using a defaultdict. This dictionary is returned together
    with the subjectivity and polarity of the text. The pattern of certain words being exclusively in AI-generated texts
    is evaluated.
    :param text: Text of articles
    :return: text subjectivity, polarity, dictionary with 10 most common high-sentiment words per text
    """
    # Load spacy pipeline
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    doc = nlp(text)

    # Add words to a list
    assessments = [assessment for assessment in doc._.blob.sentiment_assessments.assessments]
    # initialising defaultdict to keep track of words and their frequencies
    word_frequencies = defaultdict(int)
    # iterating through word list
    for words in assessments:
        # splitting up 'words' into the word and its score
        word, score = words[0], words[1]
        # selecting high-sentiment words only
        if score > 0.8 or score < -0.8:
            for str_word in word:
                word_frequencies[str_word] += 1
    # sort dictionary
    sorted_word_frequency = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)

    # words that usually only appear in the top ten high-sentiment words in AI texts
    ai_words = ['ruthless', 'brilliant', 'happy', 'legendary', 'horrific', 'beautiful', 'grim']
    # initialise the variable 'evaluation'
    evaluation = False
    # pattern 1: above words were found to only appear in the high-sentiment/frequency dictionary of AI-generated
    # articles
    for word, frequency in sorted_word_frequency:  # iterate through sorted dictionary
        # check if word from dictionary is in the words typical for AI-generated texts
        if word in ai_words:
            evaluation = True
            # exit the loop if an AI word is found in the top 10
            break

    return sorted_word_frequency, doc._.blob.polarity, doc._.blob.subjectivity, evaluation


def sentiment_ratios(text):
    """
    In this function, high sentiment ratio, the percentage of positive and negative sentences, and the distribution
    of positive versus negative sentences are found. If the results are significantly different based on whether the
    text is AI-generated or not, a new pattern is established and the text is evaluated based on that. Finally, the
    variables of this function are returned.
    """
    # Load spacy pipeline
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    doc = nlp(text)

    # initialise variable high-sentiment sentences
    high_sentiment_sentences = 0
    # initialise variable sentences
    sentences = 0
    # iterate through all sentences
    for sentence in doc.sents:
        # update to keep track of sentences
        sentences += 1
        sentence_polarity = sentence._.blob.polarity
        # select only high-sentiment sentences
        if sentence_polarity > 0.8 or sentence_polarity < -0.8:
            high_sentiment_sentences += 1
    # calculate percentage of high-sentiment sentences in relation to the entire text
    high_sentiment_ratio = high_sentiment_sentences / sentences

    # calculate percentage of positive sentences overall
    positive_sents = [1 for sentence in doc if sentence._.blob.polarity > 0]
    positive_sents_perc = sum(positive_sents) / sentences
    # pattern: AI-texts usually have a positive-sentence percentage lower than 0.7
    if positive_sents_perc < 0.7:
        evaluation_pos_sentences = True
    else:
        evaluation_pos_sentences = False

    # calculate percentage of negative sentences overall
    negative_sents = [1 for sentence in doc if sentence._.blob.polarity < 0]
    negative_sents_perc = sum(negative_sents) / sentences
    # pattern: AI-texts usually have a negative-sentence percentage lower than 0.4
    if negative_sents_perc < 0.4:
        evaluation_neg_sentences = True
    else:
        evaluation_neg_sentences = False

    # calculating the relationship of positive versus negative sentences
    pos_neg_ratio = positive_sents_perc / negative_sents_perc

    return high_sentiment_ratio, evaluation_pos_sentences, evaluation_neg_sentences, positive_sents_perc, negative_sents_perc, pos_neg_ratio


def sentiment_analysis_asent(text):
    """
    Sentiment analysis using ASEnt. It first establishes the necessary objects for spacy. It evaluates whether the
    text is AI-generated based on a pattern. The evaluation and polarity of the text are returned.
    """
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    nlp.add_pipe('asent_en_v1')
    doc = nlp(text)

    polarity = doc._.polarity
    compound_score = polarity.compound

    # pattern: in the development data, the compound score for human text was negative while for AI texts, it was
    # positive. However, this is the weakest pattern so far.
    if compound_score > 0:
        evaluation = True
    else:
        evaluation = False

    return doc._.polarity, evaluation


def discourse_analysis(text):
    """
    Using LingFeat's Discourse Analysis functions, namely Entity Density and Entity Grid features, as well as
    readability functions. It first establishes the necessary preprocessing and then calls the necessary methods,
    finally returning them.
    :param text: articles
    :return: Entity Density, Entity Grid Features, Readability Features for both parameters
    """
    # Extract discourse and readability features using LingFeat
    LingFeat = extractor.pass_text(text)
    LingFeat.preprocess()
    TraF = LingFeat.TraF_()  # Readability Features
    EnGF = LingFeat.EnGF_()  # Entity Grid Features

    return EnGF, TraF
