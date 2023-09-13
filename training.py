# import random and numpy for numerical operations, json for handling json files, pickle for serialization,
# nltk for natural language processing
import random
import numpy as np
import json
import pickle

import nltk
nltk.download('wordnet')
nltk.download('punkt')

# wordNetLemmatizer from nltk which is used for normalizing and classifying words
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# lemmatizer is initialized and content of intents.json is loaded into dict called intents
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# variables will be used later
words = []
classes = []
documents = []
ignore_letters = ['.', '!', ',', '?']

# access each element of each intent in json file that we put into the dictionary
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each pattern into words, add to wordlist
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # list of tuples, each contains a wordlist and corresponding tag
        documents.append((word_list, intent['tag']))
        # add tags to classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# normalize (base root form) words in the words list and remove chars from ignore_letters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# sort, convert to set to remove duplicates
words = sorted(set(words))
classes = sorted(set(classes))

# preprocessing data save as pickled files to use later
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# store training data
training = []
output_empty = [0] * len(classes)

# process each pattern-intent pair in documents
# create simplified and efficient way to represent the presence or absence of words from the vocab in doc or pattern
# this is so that it can be used as an input for a machine learning algorithm
for doc in documents:
    # create a bag of words (binary format)
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        if word in word_patterns:
            bag.append(int(1))  # Append 1 if word is in word_patterns
        else:
            bag.append(0)  # Append 0 if word is not in word_patterns

    # Create an output_row with one-hot encoding ( convert categorical data into numerical format
    # that can be used by machine learning algorithms)
    # create copy of output_empty as new list to ensure a fresh list for each training example
    output_row = list(output_empty)
    # doc[1] represents the intent tag associated with the current training example
    # this element corresponding to the index of the current intent tag to 1
    # marks the correct intent class for this training example
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle the training data to improve training process
random.shuffle(training)
# Convert the training list to a NumPy array
training = np.array(training)


# the x and y values that we are using to train our neural network
# x is the bag of words; binary representation of our words in the training data
train_x = list(training[:, 0])
# y is all the target labels; one-hot encoded representations of the intent classes
train_y = list(training[:, 1])

# build model
# build sequential using keras library, use 3 dense (fully connected) layers
model = Sequential()
# first layer has 128 neurons and uses ReLU (rectified linear unit) activation function
# each neuron in this layer is connected to every neuron in the previous layer (input layer in this case)
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# prevents overfitting, roughly 50% of neurons in previous layer are set to 0
model.add(Dropout(0.5))
# second layer, 64 neurons
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# softmax activation computes probability of distribution over classes
model.add(Dense(len(train_y[0]), activation='softmax'))

# stochastic gradient descent
# learning rate; step size at which model's weights are updated during training (1% of the gradients val)
# decay reduces learning rate over training
# momentum helps accelerate convergence (stabilization) by adding a fraction of the prev vector to current
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# compile the neural network with the specific configuration
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train using fit method, run for 200 epochs (one complete pass through the entire training dataset)
# batch is number of training examples that are used in one forward or backward pass of training alg
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('NNchatbot.model', hist)
print("done")