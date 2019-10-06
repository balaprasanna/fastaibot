from flask import Flask, request, jsonify, make_response
from fastai.vision import *
app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def main():
    req = request.get_json(silent=True, force=True)
    #intent_name = req["queryResult"]["intent"]["displayName"]
    print(req)
    return "hello"

@app.route("/dummy", methods=["GET"])
def dummy():
    path = untar_data(MNIST_PATH)
    data = image_data_from_folder(path)
    learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    learn.fit(1)
    return "done..."

app.run('0.0.0.0', port=9000)