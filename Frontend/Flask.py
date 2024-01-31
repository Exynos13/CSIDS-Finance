from distutils.log import debug
import numpy as np
import pandas as pd
from unicodedata import name
from io import TextIOWrapper
import csv
from flask import Flask, render_template , request, session,redirect, url_for
import matplotlib.pyplot as plt
from predictfunc import ExplainableDecisonTree
app = Flask(__name__,static_folder='static')

# Define secret key to enable session
app.secret_key = 'data janitors'
 
# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}
# Configure upload file path flask



@app.route("/")
def Home():
    return render_template("index.html")

@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        if request.method == 'POST':
            csv_file = request.files['uploaded-file']
            csv_file = TextIOWrapper(csv_file, encoding='utf-8')
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            data = []
            for row in csv_reader:
                data.append(row)
            session['data'] = data
        return render_template("index2.html")

@app.route('/predict',methods=['POST'])
def predict():
    data = session.get('data', None)
    user_data={}
    for line in data:
        name=line[0]
        line=[int(i) for i in line[1:]]
        user_data[name]=line
    user_outputs ={}
    imgs=[]
    for user in user_data:
   
    
    
        features = user_data[user]
        edt_array,value=ExplainableDecisonTree().main(features,user)
        img='./static/dtreeviz_'+str(user)+'.svg'
        imgs.append(img)
        if value[0] == 0:
            credit = 'Loan risk == GOOD '
        else:
            credit = 'Loan risk == BAD'
        
        

        output=(ExplainableDecisonTree.explainoutput(edt_array))
        user_outputs[user]=[credit]+output
    
    return render_template("predict.html",your_list=user_outputs,img=imgs,debug=True)
    


if __name__ == "__main__":
    app.run(debug=True)

    
    

