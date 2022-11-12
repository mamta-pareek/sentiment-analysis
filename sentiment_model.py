import pandas as pd

from sklearn import *

import streamlit as st

import os
import os.path
from os import path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textacy import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

import pickle
import joblib

df = pd.read_csv(
    "/home/local/ZOHOCORP/mamta-zsch897/Desktop/streamlit/amazon_cells_labelled-Copy1.txt",
    names=["review", "sentiment"],
    sep="\t",
)
# text = df.head()


reviews = df["review"].values
labels = df["sentiment"].values
reviews_train, reviews_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.2
)  # random_state=1000)


def clean_text(text):
    cleaned_text = preprocessing.make_pipeline(
        preprocessing.remove.html_tags,
        preprocessing.replace.emails,
        preprocessing.remove.punctuation,
        preprocessing.normalize.bullet_points,
        preprocessing.normalize.hyphenated_words,
        preprocessing.replace.currency_symbols,
        preprocessing.replace.phone_numbers,
        preprocessing.replace.urls,
        preprocessing.replace.emojis,
        preprocessing.normalize.whitespace,
    )
    cleaned_text1 = cleaned_text(text)

    tokenized_words = word_tokenize(str(cleaned_text1.lower()), "english")
    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words("english"):
            final_words.append(word)
    return str(final_words)


# Transform each text into a vector of word counts
vectorizer = CountVectorizer(stop_words=None, lowercase=True, preprocessor=clean_text)

x_train = vectorizer.fit_transform(reviews_train)
x_test = vectorizer.transform(reviews_test)


class singleton:
    @staticmethod
    def __init__():
        # self.model = None
        print("start")

    # Training
    def train(self):
        # if not os.path.exists("sentiment.pkl"):
        model = LogisticRegression()
        model.fit(x_train, y_train)

        # Save the model as a pickle in a file
        joblib.dump(model, "sentiment.pkl")

        print("trained")

    def load(self):
        _model = joblib.load("sentiment.pkl")
        print("loaded")
        return _model


def predict(data):
    s = singleton()
    if not os.path.exists("sentiment.pkl"):
        s.train()

    prediction = self._model.predict([data])
    return prediction


predict("beautiful plant")
print("end")

# accuracy = model.score(x_test, y_test)
# print("Accuracy: ", accuracy)


# Save the model as a pickle in a file
# joblib.dump(model, "sentiment.pkl")

# Load the model from the file
# _model = joblib.load("sentiment.pkl")
