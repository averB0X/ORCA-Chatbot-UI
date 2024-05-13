#libs
import random
import json
import pickle
import numpy as np
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# loads intents dictionary
# loads pickle files of words and classes in reading binary mode (rb)
# loads trained model
dictionary = open('intents/intents.json').read()
intents = json.loads(dictionary)
words = pickle.load(open('pkl/words.pkl', 'rb'))
classes = pickle.load(open('pkl/classes.pkl', 'rb'))
model = load_model('model/chatbotModel.h5')
lem = WordNetLemmatizer()


# <=========== functions ===========>


def cleanUpSentence(sentence):
    tokens = nltk.word_tokenize(sentence)  # tokenize sentence
    tokens = [lem.lemmatize(word) for word in tokens]  # lemmatize sentence
    return tokens

# converts sentence to a list of 0s and 1s
# the value of a user input word is stored as a '1' in the bag if it occurs in the list:words
# returns a numpy array of the bag
def bagOfWords(sentence):
    tokens = cleanUpSentence(sentence)  # tokenize sentence
    bag = [0] * len(words)  # creates a bag of 0s equal to the number of words
    for token in tokens:
        for i, word in enumerate(words):
            if word == token:
                bag[i] = 1
    return np.array(bag) 

# predicts the class (tag) where a sentence falls under
def predictClass(sentence):
    bow = bagOfWords(sentence)
    res = model.predict(np.array([bow]))[0]  # passes the numpy array of the bag of words
    ERRORTHRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERRORTHRESHOLD]  # uses the softmax activation function which returns the probability that a certain input belongs to a specific class | if the probability is 25% or lower then disregard
    
    results.sort(key=lambda x: x[1], reverse=True)  # sorts results probability in descending order
    returnList = []
    for r in results:
        returnList.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return returnList  # returns a list of intents and its probabilities

def getResponse(intentsList, intents_json):
    tag = intentsList[0]['intent']
    listOfIntents = intents_json['intents']
    for i in listOfIntents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

    # return "Sorry, I don't understand."

# chatbot loop
# print("Test running...")

# while True:
#     msg = input("User: ")
#     if msg == "debugquit":
#         exit()
#     else:
#         ints = predictClass(msg)
#         res = getResponse(ints, intents)
#         print("ORCA: " + res)