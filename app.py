from flask import Flask,request
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import linalg
import json
from chatbot import chatbot

app = Flask(__name__)

@app.route('/')
def hello_world():
    return json.dumps({'msg':"Hello World"})

@app.route('/chatbot',methods=['GET','POST'])
def chat():
    if request.method == 'POST':
        inp = request.form.get('question')
        response = chatbot(inp)
        return json.dumps(response)
    else:
        return json.dumps({'msg':'plz send post req'})


if __name__ == "__main__":
    app.run(debug=True)