# libs
import random
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import nltk
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('wordnet')
# import tensorflow as tf


def nameThisFunction():
    # iterates over the dictionary
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
                
                
# ignored characters
# words -> list of tokenized words
# tags -> tags defined from dictionary
# documents -> word (tokenized) and tag relation
exclude = ['.', ',', '!', '?']
words = []  # list for each tokenized words (words are separated from each other in a statement/phrase)
classes = []  # class or label (tags)
documents = []  # list for the combinations, where each tokenized words belong in relation to tags

# intents dictionary
# intents = json.loads(open('./intents/intents.json').read())
dictionary = open('./intents/intents.json').read()
intents = json.loads(dictionary)
nameThisFunction()

dictionary = open('./intents/intentsTechnical.json').read()
intents = json.loads(dictionary)
nameThisFunction()

dictionary = open('./intents/intentsOthers.json').read()
intents = json.loads(dictionary)
nameThisFunction()

lem = WordNetLemmatizer()
words = [lem.lemmatize(word) for word in words if word not in exclude]  # if word is not in exclude, lemmatize word
words = sorted(set(words))  # removes duplicated words
classes = sorted(set(classes))  # removes duplicate tags

# serializes each element | wb -> writing binary | outputs a pickle file (.pkl)
pickle.dump(words, open('./pkl/words.pkl', 'wb'))
pickle.dump(classes, open('./pkl/classes.pkl', 'wb'))
print("Pickle files exported.")

# MACHINE LEARNING
training = []
outputEmpty = [0] * len(classes)  # template of zeroes (0), however many classes there are

# when loop is ran, all combinations (documents) is stored in the list:training
for document in documents:
    bag = []  # for each combination(documents), creates an empty bag of words
    wordPatterns = document[0]
    wordPatterns = [lem.lemmatize(word.lower()) for word in wordPatterns]  # lemmatize each word in wordPatters, which consist of the index 0 (words) in each document element
    
    # inputs 1 or 0 into the bag of words depending whether it occurs in the pattern or not, respectively
    for word in words:
      bag.append(1) if word in wordPatterns else bag.append(0)
    # for word in words:
    #     if word in wordPatterns:
    #         bag.append(1)
    #     else:
    #         bag.append(0)
    
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)  # stores the value of bag (document[0]) and outputRow (document[1]) to training list which is either 1 or 0

random.shuffle(training)  # shuffles training data
training = np.array(training)  # converts to numpy array

# splits the array into two dimensions, x for words and y for classes
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# building the neural network
model = Sequential([
    # input layer with 128 neurons
    # input shape is dependent to the shape of the training data for x
    # activation function = rectified linear unit | if feature is determined to be significant label as 1, otherwise 0
    Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    Dropout(0.5),
    
    Dense(64, activation='relu'),
    Dropout(0.5),
    
    # output layer
    # activation function = softmax | returns the probability that a certain input belongs to a specific class (tag)
    Dense(len(trainY[0]), activation='softmax')  
])

# model = tf.keras.Sequential()  # creates a model that is layer by layer
# model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))  # input layer with 128 neurons, 
# model.add(tf.keras.layers.Dropout(0.5))  # prevents model from overfitting | randomly ignores/drops neurons temporarily each iteration | 0.5 = drops 50% of the input units/neurons
# model.add(tf.keras.layers.Dense(64, activation='relu'))  # adds a new layer with 64 neurons, allows for learning more properties/features from an input
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# optimizer | compile
# the first statement initializes an SGD optimizer with specific parameters, and the second statement compiles the model using that optimizer, along with specifying the loss function and evaluation metric.
# SGD -> stochastic gradient descent | refer to 'https://keras.io/api/optimizers/sgd/' for more information
# learning_rate=0.01 (default) | lower values means slower but more precise and stable learning
# momentum -> accelerates gradient descent in the relevant direction and dampens oscillations | 0.09 = 90% of the previous update's momentum is retained
# nesterov=True -> applies Nesterov accelerated gradient (NAG) method | helps for better performance/faster convergence
# loss='categorical-crossentropy' -> loss function used to measure the difference between the true labels and the predicted labels
# metrics=['accuracy'] -> provides a measurement for the evaluated performance of the model during training | proportion of correctly classified examples out of all examples
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True) 
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# info: 'https://keras.io/api/models/model_training_apis/'
# trains the model using the training data: 'trainX' & 'trainY'
# epochs -> number of times the entire training dataset is passed through the neural network
# batch_size -> number of samples per gradient update
# verbose -> display info during each epoch | 0 = silent, 1 = progress bar, 2 = one line per epoch
# model.save -> passes the return value of 'history' to be saved as an h5 file
print("Building model...")
history = model.fit(trainX, trainY, epochs=100, batch_size=5, verbose=1)
model.save('./model/chatbotModel.h5', history) 
print('Model created.')