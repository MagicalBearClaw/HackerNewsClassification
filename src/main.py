# -------------------------------------------------------
# Assignment 2
# Written by Michael McMahon - 26250912
# For COMP 472 Section ABJX – Summer 2020
# --------------------------------------------------------
from typing import Tuple, List, Set

import pandas as pd
import os
import time as t
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from src.HackerNewsNaiveBayesClassifier import HackerNewsNaiveBayesClassifier


# use to save vocabulary and removed words
def save_data(data: List[str], file_name: str) -> None:
    if os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name, 'w', encoding='utf-8') as writer:
        for word in data:
            writer.write(f"{word}\n")


# creates a set of stopwords form the given file.
def create_stop_word_set(file_name: str) -> Set[str]:
    stop_words: Set[str] = set()

    with open(file_name, 'r', encoding='utf-8') as reader:
        for word in reader.readlines():
            if word not in stop_words:
                stop_words.add(word.strip())

    return stop_words


start_time = t.time()

# create navie bayes classifier specific to the hacker news data set
data_set_file_name = '../data/hns_2018_2019.csv'
classifier = HackerNewsNaiveBayesClassifier(data_set_file_name)

# split data set
data_sets: Tuple[pd.DataFrame, pd.DataFrame] = classifier.split_data_set()
training_data, testing_data = data_sets

stop_words = create_stop_word_set('../data/stopwords.txt')

# get the possible labeled classes
post_types = classifier.get_classes()

##########################################
#
#         Base Experiments
#
#########################################


# baseline experiment
vocabulary, removed_words, total_class_samples = classifier.create_vocabulary(training_data)
save_data(vocabulary.keys(), '../results/baseline-vocabulary.txt')
save_data(removed_words, '../results/baseline-removed_words.txt')

model = classifier.train(vocabulary)
classifier.save_model(model, '../results/model-2018.txt')
baseline_model_vocabulary_size = model.index.size
base_line_model_results = classifier.classify_testing_data(model, testing_data, total_class_samples)
classifier.save_model_results(base_line_model_results, '../results/baseline-results.txt')

# stop-word filtering experiment
vocabulary, removed_words, total_class_samples = classifier.create_vocabulary(training_data,
                                                                              stopwords=stop_words)
save_data(vocabulary.keys(), '../results/stopword-vocabulary.txt')
save_data(removed_words, '../results/stopword-removed-words.txt')

model = classifier.train(vocabulary)
classifier.save_model(model, '../results/stopword-model.txt')
stop_word_model_results = classifier.classify_testing_data(model, testing_data, total_class_samples)
classifier.save_model_results(stop_word_model_results, '../results/stopword-results.txt')

# word length filtering experiment
vocabulary, removed_words, total_class_samples = classifier.create_vocabulary(training_data,
                                                                              cap_word_length=True)

save_data(vocabulary.keys(), '../results/baseline-word-length-vocabulary.txt')
save_data(removed_words, '../results/baseline-word-length-removed_words.txt')

model = classifier.train(vocabulary)
classifier.save_model(model, '../results/baseline-word-length-model-2018.txt')
base_line_word_length_model_results = classifier.classify_testing_data(model, testing_data, total_class_samples)
classifier.save_model_results(base_line_word_length_model_results, '../results/baseline-word-length-results.txt')

# word length filtering with stopwords experiment
vocabulary, removed_words, total_class_samples = classifier.create_vocabulary(training_data,
                                                                              stopwords=stop_words,
                                                                              cap_word_length=True)

save_data(vocabulary.keys(), '../results/stopword-word-length-vocabulary.txt')
save_data(removed_words, '../results/stopword-word-length-removed_words.txt')

model = classifier.train(vocabulary)
classifier.save_model(model, '../results/stopword-word-length-model-2018.txt')
stop_word_word_length_model_results = classifier.classify_testing_data(model, testing_data, total_class_samples)
classifier.save_model_results(stop_word_word_length_model_results, '../results/stopword-word-length-results.txt')


#########################################
#
#        Frequency Count
#
########################################

def create_row_frequency(given_model: pd.DataFrame, classes) -> pd.DataFrame:
    for index, row in given_model.iterrows():
        score = 0
        for clazz in classes:
            score += row[clazz]
        given_model.at[index, "total frequency"] = score

    return given_model


vocabulary, _, total_class_samples = classifier.create_vocabulary(training_data)

# frequency = 1 model
model = classifier.train(vocabulary)
model = create_row_frequency(model, post_types)
removed_words_model = model[~(model["total frequency"] == 1)]

freq_num_1_vocab_size = removed_words_model.index.size
freq_num_1_model_results = classifier.classify_testing_data(model[model["total frequency"] > 1], testing_data,
                                                            total_class_samples)
# frequency <= 5 model
model = classifier.train(vocabulary)
model = create_row_frequency(model, post_types)
removed_words_model = model[~(model["total frequency"] <= 5)]
freq_num_5_vocab_size = removed_words_model.index.size
freq_num_5_model_results = classifier.classify_testing_data(model[model["total frequency"] > 5], testing_data,
                                                            total_class_samples)

# frequency <= 10 model
model = classifier.train(vocabulary)
model = create_row_frequency(model, post_types)
removed_words_model = model[~(model["total frequency"] <= 10)]
freq_num_10_vocab_size = removed_words_model.index.size
freq_num_10_model_results = classifier.classify_testing_data(model[model["total frequency"] > 10], testing_data,
                                                             total_class_samples)

# frequency <= 15 model
model = classifier.train(vocabulary)
model = create_row_frequency(model, post_types)
removed_words_model = model[~(model["total frequency"] <= 15)]
freq_num_15_vocab_size = removed_words_model.index.size
freq_num_15_model_results = classifier.classify_testing_data(model[model["total frequency"] > 15], testing_data,
                                                             total_class_samples)

# frequency <= 20 model
model = classifier.train(vocabulary)
model = create_row_frequency(model, post_types)
removed_words_model = model[~(model["total frequency"] <= 20)]
freq_num_20_vocab_size = removed_words_model.index.size
freq_num_20_model_results = classifier.classify_testing_data(model[model["total frequency"] > 20], testing_data,
                                                             total_class_samples)

##########################################
#
#         Frequency Percentage
#
#########################################

vocabulary, _, total_class_samples = classifier.create_vocabulary(training_data)
freq_percentage_model = classifier.train(vocabulary)
# # frequency %= 5 model

freq_percentage_5_model = classifier.remove_words_by_frequency_percentage(freq_percentage_model, 0.05)
freq_percentage_5_model_results = classifier.classify_testing_data(freq_percentage_5_model, testing_data,
                                                                   total_class_samples)
# frequency % <= 10 model
freq_percentage_10_model = classifier.remove_words_by_frequency_percentage(freq_percentage_model, 0.10)
freq_percentage_10_model_results = classifier.classify_testing_data(freq_percentage_10_model, testing_data,
                                                                    total_class_samples)
# frequency % <= 15 model
freq_percentage_15_model = classifier.remove_words_by_frequency_percentage(freq_percentage_model, 0.15)
freq_percentage_15_model_results = classifier.classify_testing_data(freq_percentage_15_model, testing_data,
                                                                    total_class_samples)
# frequency % <= 20 model
freq_percentage_20_model = classifier.remove_words_by_frequency_percentage(freq_percentage_model, 0.20)
freq_percentage_20_model_results = classifier.classify_testing_data(freq_percentage_20_model, testing_data,
                                                                    total_class_samples)
# frequency % <= 25 model
freq_percentage_25_model = classifier.remove_words_by_frequency_percentage(freq_percentage_model, 0.25)
freq_percentage_25_model_results = classifier.classify_testing_data(freq_percentage_25_model, testing_data,
                                                                    total_class_samples)
end_time = t.time()
elapsed_time = end_time - start_time
print(f"elapsed time in {elapsed_time * 1000} ms")

##########################################
#
#         Graphs U.I
#
#########################################

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Performance Metric')


def create_metrics(given_model: pd.DataFrame) -> Tuple[float, float, float, float]:
    classes = classifier.get_classes()
    report = classification_report(given_model['class'].to_list(), given_model['prediction'].to_list(),
                                   labels=classes, output_dict=True)
    return report['accuracy'], report['macro avg']['precision'], report['macro avg']['recall'], \
           report['macro avg']['f1-score']


baseline_model_metric = create_metrics(base_line_model_results)

# frequency <= max frequency metrics
f_num_1_metrics = create_metrics(freq_num_1_model_results)
f_num_5_metrics = create_metrics(freq_num_5_model_results)
f_num_10_metrics = create_metrics(freq_num_10_model_results)
f_num_15_metrics = create_metrics(freq_num_15_model_results)
f_num_20_metrics = create_metrics(freq_num_20_model_results)


def create_list_sorted_freq_tuples():
    lists = list()
    for i in range(4):
        freq_num_list_tuples = list()
        freq_num_list_tuples.append((baseline_model_vocabulary_size, baseline_model_metric[i]))
        freq_num_list_tuples.append((freq_num_1_vocab_size, f_num_1_metrics[i]))
        freq_num_list_tuples.append((freq_num_5_vocab_size, baseline_model_metric[i]))
        freq_num_list_tuples.append((freq_num_10_vocab_size, f_num_5_metrics[i]))
        freq_num_list_tuples.append((freq_num_15_vocab_size, f_num_10_metrics[i]))
        freq_num_list_tuples.append((freq_num_20_vocab_size, f_num_15_metrics[i]))
        freq_num_list_tuples = sorted(freq_num_list_tuples, key=lambda tup: tup[0])
        lists.append(freq_num_list_tuples)

    return lists


sorted_lists = create_list_sorted_freq_tuples()

f_num_vocabulary_size = [i[0] for i in sorted_lists[0]]

f_num_accuracies = [i[1] for i in sorted_lists[0]]

f_num_precisions = [i[1] for i in sorted_lists[1]]

f_num_recall = [i[1] for i in sorted_lists[2]]

f_num_f1_score = [i[1] for i in sorted_lists[3]]

ax1.plot(f_num_vocabulary_size, f_num_accuracies, 'tab:cyan', linewidth=2, marker='o')
ax1.plot(f_num_vocabulary_size, f_num_precisions, 'tab:green', linewidth=2, marker='o')
ax1.plot(f_num_vocabulary_size, f_num_recall, 'tab:pink', linewidth=2, marker='o')
ax1.plot(f_num_vocabulary_size, f_num_f1_score, 'tab:red', linewidth=2, marker='o')
ax1.set_title('Word frequency <= max frequency')
ax1.set(xlabel='vocabulary', ylabel='performance')
ax1.label_outer()
ax1.legend(labels=["Accuracy", "Precision", "Recall", "F1 Measure"], loc="upper right")

# top % frequencies removed

f_percentage_5_metrics = create_metrics(freq_percentage_5_model_results)
f_percentage_10_metrics = create_metrics(freq_percentage_10_model_results)
f_percentage_15_metrics = create_metrics(freq_percentage_15_model_results)
f_percentage_20_metrics = create_metrics(freq_percentage_20_model_results)
f_percentage_25_metrics = create_metrics(freq_percentage_25_model_results)


def create_list_sorted_freq_percentage_tuples():
    lists = list()
    for i in range(4):
        freq_list_tuples = list()
        freq_list_tuples.append((baseline_model_vocabulary_size, baseline_model_metric[i]))
        freq_list_tuples.append((freq_percentage_model.index.size - freq_percentage_25_model.index.size,
                                 f_percentage_25_metrics[i]))
        freq_list_tuples.append((freq_percentage_model.index.size - freq_percentage_20_model.index.size,
                                 f_percentage_20_metrics[i]))
        freq_list_tuples.append((freq_percentage_model.index.size - freq_percentage_15_model.index.size,
                                 f_percentage_15_metrics[i]))
        freq_list_tuples.append((freq_percentage_model.index.size - freq_percentage_10_model.index.size,
                                 f_percentage_10_metrics[i]))
        freq_list_tuples.append((freq_percentage_model.index.size - freq_percentage_5_model.index.size,
                                 f_percentage_5_metrics[i]))

        freq_num_list_tuples = sorted(freq_list_tuples, key=lambda tup: tup[0])
        lists.append(freq_num_list_tuples)

    return lists


f_percentage_sorted_list = create_list_sorted_freq_percentage_tuples()

f_percentage_vocabulary_sizes = [i[0] for i in f_percentage_sorted_list[0]]

f_percentage_accuracies = [i[1] for i in f_percentage_sorted_list[0]]

f_percentage_precisions = [i[1] for i in f_percentage_sorted_list[1]]

f_percentage_recall = [i[1] for i in f_percentage_sorted_list[2]]

f_percentage_f1_score = [i[1] for i in f_percentage_sorted_list[3]]

ax2.plot(f_percentage_vocabulary_sizes, f_percentage_accuracies, 'tab:cyan', linewidth=2, marker='o')
ax2.plot(f_percentage_vocabulary_sizes, f_percentage_precisions, 'tab:green', linewidth=2, marker='o')
ax2.plot(f_percentage_vocabulary_sizes, f_percentage_recall, 'tab:pink', linewidth=2, marker='o')
ax2.plot(f_percentage_vocabulary_sizes, f_percentage_f1_score, 'tab:red', linewidth=2, marker='o')
ax2.set_title('Top x frequency percentage removed')
ax2.set(xlabel='vocabulary', ylabel='performance')
ax2.legend(labels=["Accuracy", "Precision", "Recall", "F1 Measure"], loc="upper right")

plt.show()
