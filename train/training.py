import random
import json
import pickle  # for serialization
import numpy as np
import tensorflow as tf
    # from tensorflow import keras
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Dense, Activation, Droupout
    # from tensorflow.keras.optimizers import SGD

import nltk  # natural language toolkit
# nltk.download('punkt')  # resolves "Resource punkt not found."
# nltk.download('wordnet')  # resolves "Resource wordnet not found."
from nltk.stem import WordNetLemmatizer  # optimization | groups same words or reduces similar words to its stem


lemmatizer = WordNetLemmatizer()

# loads intents library
intents = json.loads(open('./intents/intents.json').read())
# intentsAdmission = json.load(open('./intents/admissionIntents.json').read())
# intentsRegistrar = json.load(open('./intents/registrarIntents.json').read())
# intentsCashier = json.load(open('./intents/cashierIntents.json').read())

words = []  # list for each tokenized words (words are separated from each other in a statement/phrase)
classes = []  # list of tags
documents = []  # list for the combinations, where each tokenized words belong in relation to tags
ignoreLetters = ['?', '!', '.', ',']

# iterate over the intents library
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]  # lemmatize word of word is not in list:ignoreLetters
words = sorted(set(words))  # remove dupes
classes = sorted(set(classes))

# save into output pickle file
pickle.dump(words, open('./pkl/words.pkl', 'wb'))
pickle.dump(classes, open('./pkl/classes.pkl', 'wb'))


# MACHINE LEARNING ////////////////////////////////////////////////////////////////////////
# sets the value of the word to either 1 or 0 depending whether it occurs in the pattern or not, respectively
training = []
outputEmpty = [0] * len(classes)

# when loop is ran, all combination data or document is stored in the list:training
for document in documents:
    bag = []  # for each combination(documents), creates an empty bag of words
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)  # checks if word occurs in a pattern, append 1 if word matches pattern, otherwise 0

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)



random.shuffle(training)
training = np.array(training)  # converts to numpy array

# splits the array into two dimensions, x and y
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# trainX = list(training[:,0])
# trainY = list(training[:,1])

# trainX = np.array([item[0] for item in training])
# trainY = np.array([item[1] for item in training])

# building the neural network model (I don't understand shit)
model = tf.keras.Sequential()  # creates a model that is layer by layer
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))  # input layer with 128 neurons, 
model.add(tf.keras.layers.Dropout(0.5))  # prevents model from overfitting | randomly ignores/drops neurons temporarily each iteration | 0.5 = drops 50% of the input units/neurons
model.add(tf.keras.layers.Dense(64, activation='relu'))  # adds a new layer with 64 neurons, allows for learning more properties/features from an input
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))
    # model = Sequential()
    # model.add(Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(len(trainY[0]), activation='softmax'))

# neural network theory shits that I don't understand
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
model.save('./model/chatbotModel.h5', hist)
print('Done')