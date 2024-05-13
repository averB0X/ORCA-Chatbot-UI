import json

from flask import Flask, render_template, request, jsonify
from chatbotv1 import predictClass
from chatbotv1 import getResponse

app = Flask(__name__)

@app.route("/")
def index_get():
    return render_template('index.html')


@app.post("/predict")
def predict():
    dictionary = open('intents/intents.json').read()
    intents = json.loads(dictionary)
    msg = input("User: ")
    if msg == "debugquit":
        exit()
    else:
        ints = predictClass(msg)
        res = getResponse(ints, intents)
        return jsonify(res)
        
        
if __name__ == '__main__':
    app.run(host="127.0.0.1", port="8080", debug=True)