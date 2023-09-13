import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
# import pre-trained model with tensorflow keras
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')


# tokenizes and lemmatizes the words in a sentence and returns them as a list of words
def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# given a sentence, return a binary BOW representation
def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


# convert sentence to BOW, use loaded NN model to predict intent of the sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: [1], reverse=True)
    return_lis = []
    for r in result:
        return_lis.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_lis
