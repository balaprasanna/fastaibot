from flask import Flask, request, jsonify, make_response
from fastai.core import *
from fastai.vision import *
from fastai.metrics import error_rate
import io
from contextlib import redirect_stdout
import subprocess

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
    
    f = io.StringIO()
    with redirect_stdout(f):
        learn.fit(1)
    out = f.getvalue()
    print(out)
    return out


@app.route("/pull", methods=["GET"])
def restart():
    print( run_in_term( "git pull" ) )


def run_in_term(command):
    with open("/tmp/subprocessout.txt", "w") as f:
        out = subprocess.call( command.split() , shell=True, stdout=f)
        if out != 0 :
            raise Exception("Non zero exit...")
    return open("/tmp/subprocessout.txt", "r").read()

app.run('0.0.0.0', port=9000)