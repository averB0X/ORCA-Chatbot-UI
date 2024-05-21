import json

from flask import Flask, render_template, request, jsonify
from chatbotv2 import predictClass
from chatbotv2 import getResponse

app = Flask(__name__)

@app.route("/")
def index_get():
    return render_template("index.html")

@app.post("/predict")
def predict():
    msg = request.get_json().get("message")
    ints = predictClass(msg)
    res = getResponse(ints)
    message = {"answer": res}
    return jsonify(message)
        
if __name__ == "__main__":
    app.run(host="127.0.0.1", port="8080", debug=True)