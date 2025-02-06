import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import random as rd
import os
from flask import Flask, request, jsonify, render_template, url_for
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = pickle.load(open('D:\\Customer Segmentation Final1\\Customer Segmentation\\xgbmodel.pk2','rb'))
scale = pickle.load(open('D:\\Customer Segmentation Final1\\Customer Segmentation\\scale.pk2','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST","GET"])

def predict():
    input_feature = [float(x) for x in request.form.values()]
    print("Input Feature:", input_feature)
    features_values = [np.array(input_feature)]
    names = ['Sex', 'Martial status', 'Age', 'Education', 'Income', 'Occupation', 'Settlement size']
    data = pandas.DataFrame(features_values, columns=names)
    scale = StandardScaler()
    scaled_data = scale.fit_transform(data)
    print("Scaled Data:", scaled_data)
    prediction = model.predict(data)
    print("Raw Prediction:", prediction)
    # Determine prediction text based on prediction
    if prediction == 0:
        prediction_text = "Not a potential customer"
        return render_template("not_potential.html", prediction_text=prediction_text)

    elif prediction == 1:
        prediction_text = "Potential customer"
        return render_template("potential.html", prediction_text=prediction_text)

    else:
        prediction_text = "Highly potential customer"
        return render_template("highly_potential.html", prediction_text=prediction_text)

    
    # Render template with prediction text
    #return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True,port=5000)
    #app.run(ssl_context=('cert.pem', 'key.pem'),debug=True,port=3000)