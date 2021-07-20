# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 19:15:55 2021

@author: user
https://www.youtube.com/watch?v=8vNBW98LbfI&list=PLZoTAELRMXVNKtpy0U_Mx9N26w8n0hIbs&index=3
"""
 
from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return"Welcome all"
    
@app.route('/predict')
def predict_note_authentication():
   
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
  
    
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    
    return "the predicted value is: "+str(prediction)
    
@app.route('/predict_file',methods=["POST"])
def predict_note_authentication_testcsv():
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    
    test_df=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(test_df)
    
    return "the predicted value is: "+str(list(prediction))
    
    
    
    

if __name__=='__main__':
    app.run()
    ## or we can use this wiht specific port number app.run(host='0.0.0.0',port=8000))

