# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 19:15:55 2021

@author: user
https://www.youtube.com/watch?v=ipFUANeStYE&list=PLZoTAELRMXVNKtpy0U_Mx9N26w8n0hIbs&index=2&t=1231s
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return"Welcome all"
    
@app.route('/predict')
def predict_note_authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    
    return "the predicted value is: "+str(prediction)
    
@app.route('/predict_file',methods=["POST"])
def predict_note_authentication_testcsv():
    test_df=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(test_df)
    
    return "the predicted value is: "+str(list(prediction))
    
    
    
    

if __name__=='__main__':
    app.run()

