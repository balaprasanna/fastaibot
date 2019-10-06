from flask import Flask 

app = Flask(__name__)

@app.route("/")
def main(): return "hello"

app.run('0.0.0.0', port=9000)
