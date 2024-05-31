# Laura de Boer, Dertje Roggeveen, Roshana Vegter, Julian Paagman
# Group 2
# nlp.py

from morphology import *
from syntax import *


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
