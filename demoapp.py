from flask import Flask, request, jsonify, make_response
from fastai.core import *
from fastai.vision import *
from fastai.metrics import error_rate
app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def main():
    req = request.get_json(silent=True, force=True)
    #intent_name = req["queryResult"]["intent"]["displayName"]
    print(req)
    return "hello"

@app.route("/dummy", methods=["GET"])
def dummy():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path)
    learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    learn.fit(1)
    return "done..."

app.run('0.0.0.0', port=9000)