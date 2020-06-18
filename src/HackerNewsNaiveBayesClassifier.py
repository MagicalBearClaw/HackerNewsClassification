# -------------------------------------------------------
# Assignment 2
# Written by Michael McMahon - 26250912
# For COMP 472 Section – Summer 2020
# -------------------------------------------------------
import math
import operator
from typing import Dict, Set

import pandas as pd
import os
from nltk.tokenize import RegexpTokenizer


class HackerNewsNaiveBayesClassifier:
    def __init__(self, file_name: str):
        self._data_frame = pd.read_csv(file_name)
        self._classes = self._data_frame["Post Type"].unique()
        self._tokenizer = RegexpTokenizer("[\w']+")
        self._removal_tokenizer = RegexpTokenizer('[^\w]|[| \d]+')
        self._space_tokenizer = RegexpTokenizer('[^\s]+')
        self._default_class_count_dict = self.__create_default_class_count_dict()

    def __create_default_class_count_dict(self) -> Dict[str, int]:
        class_count_dict: Dict[str, int] = dict()
        for c in range(len(self._classes)):
            class_count_dict[self._classes[c]] = 0

        return class_count_dict

    def save_model(self, model: pd.DataFrame, file_name: str) -> None:
        if os.path.exists(file_name):
            os.remove(file_name)

        with open(file_name, 'w', encoding='utf-8') as writer:
            current_index = 0
            for index, row in model.iterrows():
                line = f"{current_index}  {index}"
                for clazz in self._classes:
                    class_prob = f"{clazz}_prob"
                    line += f"  {row[clazz]}  {row[class_prob]}"
                current_index += 1
                writer.write(f"{line}\n")

    def save_model_results(self, model: pd.DataFrame, file_name: str) -> None:
        if os.path.exists(file_name):
            os.remove(file_name)

        with open(file_name, 'w', encoding='utf-8') as writer:

            for index, row in model.iterrows():
                line = f"{index}  {row['title']}  {row['prediction']}"
                for clazz in self._classes:
                    line += f"  {row[clazz]}"

                line += f"  {row['class']}  {row['label']}"
                writer.write(f"{line}\n")

    def split_data_set(self):
        self._data_frame["Created At"] = self._data_frame["Created At"].str.strip()
        training_data = self._data_frame.loc[self._data_frame["Created At"].str.startswith('2018')]
        testing_data = self._data_frame.loc[self._data_frame["Created At"].str.startswith('2019')]
        return training_data, testing_data

    def create_vocabulary(self, data: pd.DataFrame, stopwords: Set[str] = set(), cap_word_length: bool = False):
        removed_words: set = set()
        vocabulary: Dict[str, Dict[str, int]] = dict()
        total_class_samples = dict()
        total_class_words = dict()

        for index, row in data.iterrows():
            row['Title'] = row['Title'].lower()
            words = self._tokenizer.tokenize(row["Title"])
            words_to_remove = self._removal_tokenizer.tokenize(row["Title"])

            post_type = row["Post Type"]
            if post_type not in total_class_samples:
                total_class_samples[post_type] = 0

            total_class_samples[post_type] += 1

            for word in words_to_remove:
                separated_words = self._space_tokenizer.tokenize(word)
                for w in separated_words:
                    w = w.strip()
                    if w not in removed_words:
                        removed_words.add(w)

            for word in words:
                word = word.strip()
                if word in removed_words:
                    continue

                if cap_word_length:
                    if len(word) <= 2 or len(word) >= 9:
                        removed_words.add(word)
                        continue

                if word in stopwords:
                    removed_words.add(word)
                    continue

                if word not in vocabulary:
                    vocabulary[word] = self._default_class_count_dict.copy()
                    total_class_words[post_type] = 0

                freq_dic = vocabulary[word]
                freq_dic[post_type] += 1
                total_class_words[post_type] += 1

        return vocabulary, removed_words, total_class_samples, total_class_words

    def train(self, vocabulary_frequencies: Dict[str, Dict[str, int]],
              total_class_words: Dict[str, int], smoothing: float = 0.5) -> pd.DataFrame:

        model = pd.DataFrame.from_dict(vocabulary_frequencies).T
        model.sort_index(inplace=True)
        model[self._classes] += smoothing

        for index, row in model.iterrows():
            for clazz in self._classes:
                probability = row[clazz] / (total_class_words[clazz] + (smoothing * model.size))
                new_class = f"{clazz}_prob"

                if new_class not in model.columns:
                    model[new_class] = 0.0

                model.at[index, new_class] = math.log10(probability)

        return model

    def classify_testing_data(self, model: pd.DataFrame, testing_data: pd.DataFrame,
                              total_class_samples: Dict[str, int]) -> pd.DataFrame:

        result_data = list()
        total_class_samples_count = 0
        for sample in total_class_samples:
            total_class_samples_count += total_class_samples[sample]

        for index, row in testing_data.iterrows():
            class_scores = dict()
            row['Title'] = row['Title'].lower()
            words = self._tokenizer.tokenize(row["Title"])
            words_to_remove = self._removal_tokenizer.tokenize(row["Title"])
            removed_words: set = set()

            for word in words_to_remove:
                separated_words = self._space_tokenizer.tokenize(word)
                for w in separated_words:
                    w = w.strip()
                    if w not in removed_words:
                        removed_words.add(w)

            for word in words:
                word = word.strip()
                if word in removed_words:
                    continue

                for clazz in self._classes:
                    score = 0
                    if word in model.index:
                        score = model.at[word, f"{clazz}_prob"]

                    if clazz not in class_scores:
                        class_scores[clazz] = 0

                    class_scores[clazz] += score

            result_instance_data = list()
            result_instance_data.append(row['Title'])
            for clazz in self._classes:
                class_scores[clazz] += math.log10(total_class_samples[clazz] / total_class_samples_count)

            prediction_string = "wrong"
            predicted_class = max(class_scores.items(), key=operator.itemgetter(1))[0]
            if predicted_class == row['Post Type']:
                prediction_string = 'right'

            result_instance_data.append(predicted_class)

            for clazz in self._classes:
                result_instance_data.append(class_scores[clazz])

            result_instance_data.append(row['Post Type'])
            result_instance_data.append(prediction_string)

            result_data.append(result_instance_data)

        columns = list()
        columns.append("title")
        columns.append("prediction")

        for clazz in self._classes:
            columns.append(clazz)

        columns.append('class')
        columns.append('label')

        results = pd.DataFrame(result_data, columns=columns)
        return results
