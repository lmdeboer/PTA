# Laura de Boer, Dertje Roggeveen, Roshana Vegter, Julian Paagman
# Group 2
# nlp.py

from morphology import *
from syntax import *


def syntax(text_ai, syntax_file):
    common_chunks_ai = Counter(get_chunks(text_ai))
    syntax_file.write("10 Most Common Noun Chunks in text:\n")
    for chunk, frequency in common_chunks_ai.most_common(10):
        syntax_file.write(f"{chunk}: {frequency}\n")

    common_proper_noun_chunks_ai = Counter(get_chunks(text_ai, filter_root='PROPN'))
    syntax_file.write("\n10 Most Common Noun Chunks with Proper Noun Roots in text:\n")
    for chunk, frequency in common_proper_noun_chunks_ai.most_common(10):
        syntax_file.write(f"{chunk}: {frequency}\n")

    tags_ai = []
    for word in text_ai:
        tags_ai.append(word.pos_)
    syntax_file.write("10 Most frequent POS tags in text: " +
                      str(count_tags(tags_ai)) + "\n")

    syntax_file.write("Most common specified dependency in text: " +
                      str(most_frequent_asked_dependency(text_ai, 'ADJ')) + "\n")

    syntax_file.write("----------------------------------------------------------\n")
    syntax_file.write("\n")


def semantics(text):
    pass


def pragmatic(text):
    pass


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

    semantics(human_text)
    semantics(ai_text)

    pragmatic(human_text)
    pragmatic(ai_text)


if __name__ == '__main__':
    main()
