"""**Chatbot.py**"""

"""The chatbot for the actual use of the network trained in our training script"""
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.load(open('intents.json'))

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    """
    Turns a sentence into an array of individual words
    :param sentence: A string with multiple words in it
    :return: An array of one word link strings
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """
    Jumbles the words from the user input into a bag to be thrown through the network
    :param sentence: The complete user input sentence
    :return: The array of words being used by the bag of random selection
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """
    Returns the categorized class type of the given input
    :param sentence: A string from a user input
    :return: A list of intents tag types that are present
    """
    bow = bag_of_words(sentence)
    model_res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(model_res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """
    Gives a string to be printed back to the console of the input sentence
    :param intents_list: The response class type calculated by the network
    :param intents_json: Training JSON data file
    :return: A string with a random response from the json file
    """
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# As soon as tensorflow has booted itself, this sentence will print to greet the user
print('Hello! My name is Ellie. I am here to answer your queries about the University. How may I help you today?')

# once the user is input here, the prediction and response calculator defines the next response!
while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    # print (ints)
    if ints[0]['intent'] == "goodbye":
      print ("Goodbye! Have a nice day!")
      break
    print(res)