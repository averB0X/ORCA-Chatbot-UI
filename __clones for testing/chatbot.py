import random
import json
import pickle
import numpy as np

import nltk  # natural language toolkit
# nltk.download('punkt')  # resolves "Resource punkt not found."
# nltk.download('wordnet')  # resolves "Resource wordnet not found."
from nltk.stem import WordNetLemmatizer  # optimization | groups same words
from keras.models import load_model


lemmatizer = WordNetLemmatizer()

# loads intents library and loads pickle files in reading binary mode (rb)
intents = json.loads(open('intents/intents.json').read())
words = pickle.load(open('pkl/words.pkl', 'rb'))
classes = pickle.load(open('pkl/classes.pkl', 'rb'))
model = load_model('model/chatbotModel-admissionDemo.h5')

def cleanUpSentence(sentence):
    sentenceWords = nltk.word_tokenize(sentence)
    sentenceWords = [lemmatizer.lemmatize(word) for word in sentenceWords]
    return sentenceWords

def bagOfWords(sentence):
    sentenceWords = cleanUpSentence(sentence)
    bag = [0] * len(words)
    for w in sentenceWords:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predictClass(sentence):
    bow = bagOfWords(sentence)
    res = model.predict(np.array([bow]))[0]
    ERRORTHRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERRORTHRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    returnList = []
    for r in results:
        returnList.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return returnList

def getResponse(intentsList, intents_json):
    tag = intentsList[0]['intent']
    listOfIntents = intents_json['intents']
    for i in listOfIntents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# chatbot loop
print("Test running...")

while True:
    msg = input("")
    ints = predictClass(msg)
    res = getResponse(ints, intents)
    print(res)