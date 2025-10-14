import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas

app = Flask(__name__) # starting point from where the application starts running
regmodel = pickle.load(open('regmodel.pkl','rb')) # opening the file in the read binary mode and loading into regmodel to use as our model
scalar=pickle.load(open('scaling.pkl','rb'))

# base path of the app i.e the very first URL a user goes to
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST']) # this is a POST API request cause the app here captures some input provided and sends that to the model which then gives some output
def predict_api():
    data = request.json['data'] # request is a global object provided by Flask that has all the request metadata. json is the attribute on the request object that attempts to parse this input as a JSON body
    # the line of code assumes that the data is stored under a key called 'data'
    print(data) # this will be of the form of key value pairs
    # we need to convert user data into a 2D array or DataFrame with the correct features because that's the format the machine learning model is designed to accept for both training and prediction.
    print(np.array(list(data.values())).reshape(-1,1)) 
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0]) # since this is a 2d array
    return jsonify(output[0])

# runs the whole thing and having debug = true allows future debugging
if __name__=="__main__":
    app.run(debug=True)
