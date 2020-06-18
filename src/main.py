# -------------------------------------------------------
# Assignment 2
# Written by Michael McMahon - 26250912
# For COMP 472 Section – Summer 2020
# --------------------------------------------------------
from typing import Tuple, List, Set

import pandas as pd
import os
import time as t

from src.HackerNewsNaiveBayesClassifier import HackerNewsNaiveBayesClassifier


def save_data(data: List[str], file_name: str) -> None:
    if os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name, 'w', encoding='utf-8') as writer:
        for word in data:
            writer.write(f"{word}\n")


def create_stop_word_set(file_name: str) -> Set[str]:
    stop_words: Set[str] = set()

    with open(file_name, 'r', encoding='utf-8') as reader:
        for word in reader.readlines():
            if word not in stop_words:
                stop_words.add(word.strip())

    return stop_words


start_time = t.time()
data_set_file_name = '../data/test.csv'
classifier = HackerNewsNaiveBayesClassifier(data_set_file_name)

data_sets: Tuple[pd.DataFrame, pd.DataFrame] = classifier.split_data_set()
training_data, testing_data = data_sets

# baseline experiment
vocabulary, removed_words, total_class_samples, total_class_words = classifier.create_vocabulary(training_data)

save_data(vocabulary.keys(), '../results/baseline-vocabulary.txt')
save_data(removed_words, '../results/baseline-removed_words.txt')

model = classifier.train(vocabulary, total_class_words)
classifier.save_model(model, '../results/model-2018.txt')
base_line_model_results = classifier.classify_testing_data(model, testing_data, total_class_samples)
classifier.save_model_results(base_line_model_results, '../results/baseline-results.txt')

# stop-word filtering experiment
stop_words = create_stop_word_set('../data/stopwords.txt')
vocabulary, removed_words, total_class_samples, total_class_words = classifier.create_vocabulary(training_data,
                                                                                                 stopwords=stop_words)
save_data(vocabulary.keys(), '../results/stopword-vocabulary.txt')
save_data(removed_words, '../results/stopword-removed-words.txt')

model = classifier.train(vocabulary, total_class_words )
classifier.save_model(model, '../results/stopword-model.txt')
stop_word_model_results = classifier.classify_testing_data(model, testing_data, total_class_samples)
classifier.save_model_results(stop_word_model_results, '../results/stopword-results.txt')

# word length filtering experiment
vocabulary, removed_words, total_class_samples, total_class_words = classifier.create_vocabulary(training_data,
                                                                                                 cap_word_length=True)

save_data(vocabulary.keys(), '../results/baseline-word-length-vocabulary.txt')
save_data(removed_words, '../results/baseline-word-length-removed_words.txt')

model = classifier.train(vocabulary, total_class_words)
classifier.save_model(model, '../results/baseline-word-length-model-2018.txt')
base_line_word_length_model_results = classifier.classify_testing_data(model, testing_data, total_class_samples)
classifier.save_model_results(base_line_word_length_model_results, '../results/baseline-word-length-results.txt')


# word length filtering with stopwords experiment
vocabulary, removed_words, total_class_samples, total_class_words = classifier.create_vocabulary(training_data,
                                                                                                 stopwords=stop_words,
                                                                                                 cap_word_length=True)

save_data(vocabulary.keys(), '../results/stopword-word-length-vocabulary.txt')
save_data(removed_words, '../results/stopword-word-length-removed_words.txt')

model = classifier.train(vocabulary, total_class_words)
classifier.save_model(model, '../results/stopword-word-length-model-2018.txt')
stop_word_word_length_model_results = classifier.classify_testing_data(model, testing_data, total_class_samples)
classifier.save_model_results(stop_word_word_length_model_results, '../results/stopword-word-length-results.txt')


end_time = t.time()
elapsed_time = end_time - start_time
print(f"elapsed time in {elapsed_time * 1000} ms")
