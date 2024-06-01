# Laura de Boer, Dertje Roggeveen, Roshana Vegter, Julian Paagman
# Group 2
# syntax.py

from collections import defaultdict
from spacytextblob.spacytextblob import SpacyTextBlob
from lingfeat import extractor
import asent
import spacy

def sentiment_analysis_tb(human_text, ai_text):
    """
    Sentiment analysis using TextBlob. It first establishes the necessary objects for spacy and assigns the human
    text and the AI text different objects. Moreover, the 10 most common high-sentiment words per text (Human/AI)
    are found by using a defaultdict. These two dictionaries are returned together with the subjectivity and polarity
    of each text.
    :param human_text: Text of human articles
    :param ai_text: text of AI articles
    :return: AI/Human text subjectivity, polarity, dictionary with 10 most common high-sentiment words per text
    """
    # Load spacy pipeline
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')

    doc_h = nlp(human_text)
    doc_ai = nlp(ai_text)

    human_assessments = [assessment for assessment in doc_h._.blob.sentiment_assessments.assessments]
    human_word_frequencies = defaultdict(int)
    for word in human_assessments:
        human_word, human_score = word[0], word[1]
        if human_score > 0.8 or human_score < -0.8:
            for str in human_word:
                human_word_frequencies[str] += 1
    sorted_h_word_frequency = sorted(human_word_frequencies.items(), key=lambda x: x[1], reverse=True)


    ai_assessments = [assessment for assessment in doc_ai._.blob.sentiment_assessments.assessments]
    ai_word_frequencies = defaultdict(int)
    for word in ai_assessments:
        ai_word, ai_score = word[0], word[1]
        if ai_score > 0.8 or ai_score < -0.8:
            for str in ai_word:
                ai_word_frequencies[str] += 1
    sorted_ai_word_frequency = sorted(ai_word_frequencies.items(), key=lambda x: x[1], reverse=True)

    return sorted_h_word_frequency, sorted_ai_word_frequency, doc_h._.blob.polarity, doc_h._.blob.subjectivity, doc_ai._.blob.polarity, doc_ai._.blob.subjectivity

def sentiment_analysis_asent(human_text, ai_text):
    """
    Sentiment analysis using ASEnt. It first establishes the necessary objects for spacy and assigns the human and AI
    text different objects. It then returns the polarity of both texts.
    """
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    nlp.add_pipe('asent_en_v1')

    doc_h = nlp(human_text)
    doc_ai = nlp(ai_text)

    return doc_h._.polarity, doc_ai._.polarity


def discourse_analysis(human_text, ai_text):
    """
    Using LingFeat's Discourse Analysis functions, namely Entity Density and Entity Grid features, as well as readability
    functions. It first establishes the necessary preprocessing and then calls the necessary methods, finally returning
    them.
    :param human_text: Human articles
    :param ai_text: AI articles
    :return: Entity Density, Entity Grid Features, Readbility Features for both parameters
    """
    # Extract discourse features using LingFeat
    LingFeat_AI = extractor.pass_text(ai_text)
    LingFeat_H = extractor.pass_text(human_text)

    LingFeat_AI.preprocess()
    LingFeat_H.preprocess()

    TraF_AI = LingFeat_AI.TraF_() # Readability Features
    TraF_H = LingFeat_H.TraF_() # Readability Features

    EnDF_AI = LingFeat_AI.EnDF_()  # Entity Density Features
    EnGF_AI = LingFeat_AI.EnGF_()  # Entity Grid Features

    EnDF_H = LingFeat_H.EnDF_()  # Entity Density Features
    EnGF_H = LingFeat_H.EnGF_()  # Entity Grid Features


    return EnDF_AI, EnGF_AI, TraF_AI, EnDF_H, EnGF_H, TraF_H
