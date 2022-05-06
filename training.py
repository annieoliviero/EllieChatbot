"""**Training.py**"""
"""

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Word processor that understands the difference between "Hi" and "Hello"
lemmatizer = WordNetLemmatizer()

# noinspection PyTypeChecker
# Reads the JSON file
intents = json.load(open('intents.json'))

# The words and classes that will be categorized by the neural network
words = []
classes = []
documents = []
# The characters we don't want to have to train the network on, otherwise there will be confusion
ignore_letters = ['?', '!', ',', '.']

# Loop through our training data and convert it into tokenized pieces in an array for training purposes
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in a given pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Assign a type to the tokenized list
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append((intent['tag']))

# Lemmatize the words so that words like 'Hi' and 'Hello' are categorized the same (e.g. greetings)
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

# Here are saved the words and classes organized in the training part of the model
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

# Since the network understands numbers instead of words, we are creating a bag of words for the network to choose from
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle the data to increase our accuracy
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build the neural network
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Now we train the model
history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Calculates the scores, the accuracy should be 1 since the loss is very low
score = model.evaluate(train_x, train_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Export the model for use by the chatbot in a h5 file
model.save('chatbot_model.h5', history)
print('Done training!')

