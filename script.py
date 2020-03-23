# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:22:04 2020

@author: Chinmay
"""

from flask import Flask, render_template, request, redirect
import numpy as np
import pickle




# prediction function 
def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1,-1) 
    with open('rf_classifier.pickle', 'rb') as f:
        loaded_model = pickle.load(f)
    result = loaded_model.predict(to_predict) 
    return result[0] 



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())  
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='Loan Can be granted'
        else: 
            prediction ='Loan Cannot be granted'            
        return render_template("result.html", prediction = prediction)
    


if __name__ == "__main__":
    app.run()