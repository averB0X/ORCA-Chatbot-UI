import json

from flask import Flask, render_template, request, jsonify
from chatbotv1 import predictClass
from chatbotv1 import getResponse

app = Flask(__name__)

dictionary = open("intents/intents.json").read()
intents = json.loads(dictionary)

@app.route("/")
def index_get():
    return render_template("index.html")

@app.post("/predict")
def predict():
    msg = request.get_json().get("message")
    ints = predictClass(msg)
    res = getResponse(ints, intents)
    message = {"answer": res}
    print(message)
    return jsonify(message)
        
if __name__ == "__main__":
    app.run(host="127.0.0.1", port="8080", debug=True)