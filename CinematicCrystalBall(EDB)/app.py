from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)
from flask_pymongo import PyMongo
from flask import Flask,jsonify
import pandas as pd
import numpy as np
import os
import collections
import pickle
import warnings
warnings.simplefilter('ignore')
from keras.models import load_model
#import scrape_mars

# Create an instance of Flask
app = Flask(__name__)

# Load the Randon Forest model pickle
rf_model_pkl = open('Resources/randon_forest_classifier_movie.pkl', 'rb')
rf_model = pickle.load(rf_model_pkl)

# Load the logistic regression model pickle
lr_model_pkl = open('Resources/logistic_regression_classifier_movie.pkl', 'rb')
lr_model = pickle.load(lr_model_pkl)

# Load the NN model 
nn_model = load_model("Resources/dense_movie_classifier.h5")

# read movie_final file to build the input layout for the ML model.predict
df = pd.read_csv(os.path.join("Resources", "movies_final.csv"))

# remove target from input
data_df = df.drop("Profitable", axis=1)

# get list of features
column_index = list(data_df.columns.values)
column_index

# convert list to dataframe and name the column
column_index_df = pd.DataFrame(column_index)
column_index_df.columns = ['features']

# extract featues names from column heading and create two new columns of category and feature
column_index_df["category"] = column_index_df["features"].str.split('_').str[0]
column_index_df["feature"] = column_index_df["features"].str.split('_').str[-1]

# Drop column name
column_index = column_index_df[["category", "feature"]]

# Use PyMongo to establish Mongo connection
mongo = PyMongo(app, uri="mongodb://localhost:27017/movies_predictor")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/map')
def map():
    return render_template('price.html')

@app.route('/crystal', methods = ["GET","POST"])
def crystal():
    if request.method == "POST":
        # convert the data frame to a dictionay. we are using defaultdict as feature can be
        # duplicate (an actor can also be director)
        # convert all feature to key and default it with value 0
        result = collections.defaultdict(lambda: collections.defaultdict(int))
        for idx, rows in column_index.iterrows():
            result[rows['category']][rows['feature']] = 0     

        inputgenres = []
        inputdirector = []
        inputactor = []
        

        # Get user entered Genres
        if request.form["genInput1"]:
            Genres1 = request.form["genInput1"]
            result['genres'][Genres1] = 1
            inputgenres.extend(Genres1)
        

        if request.form["genInput2"]:
            Genres2 = request.form["genInput2"]
            result['genres'][Genres2] = 1
            inputgenres.extend(Genres2)
        
        if request.form["genInput3"]:
            Genres3 = request.form["genInput3"]
            result['genres'][Genres3] = 1
            inputgenres.extend(Genres3)
        
        if request.form["genInput4"]:
            Genres4 = request.form["genInput4"]
            result['genres'][Genres4] = 1
            inputgenres.extend(Genres4)
        
        if request.form["genInput5"]:
            Genres5 = request.form["genInput5"]
            result['genres'][Genres5] = 1
            inputgenres.extend(Genres5)
        
        # Get user entered Directors
        if request.form["dirInput1"]:
            Director1 = request.form["dirInput1"]
            result['crew'][Director1] = 1
            inputdirector.extend(Director1)
        
        if request.form["dirInput2"]:
            Director2 = request.form["dirInput2"]
            result['crew'][Director2] = 1
            inputdirector.extend(Director2)
        
        if request.form["dirInput3"]:
            Director3 = request.form["dirInput3"]
            result['crew'][Director3] = 1
            inputdirector.extend(Director3)

        if request.form["dirInput4"]:
            Director4 = request.form["dirInput4"]
            result['crew'][Director4] = 1
            inputdirector.extend(Director4)
        
        if request.form["dirInput5"]:
            Director5 = request.form["dirInput5"]
            result['crew'][Director5] = 1
            inputdirector.extend(Director5)
        
        # Get user entered Actors
        if request.form["actInput1"]:
            Actor1 = request.form["actInput1"]
            result['actor'][Actor1] = 1
            inputactor.extend(Actor1)
        
        if request.form["actInput2"]:
            Actor2 = request.form["actInput2"]
            result['actor'][Actor2] = 1
            inputactor.extend(Actor2)
        
        if request.form["actInput3"]:
            Actor3 = request.form["actInput3"]
            result['actor'][Actor3] = 1
            inputactor.extend(Actor3)
        
        if request.form["actInput4"]:
            Actor4 = request.form["actInput4"]
            result['actor'][Actor4] = 1
            inputactor.extend(Actor4)
        
        if request.form["actInput5"]:
            Actor5 = request.form["actInput5"]
            result['actor'][Actor5] = 1
            inputactor.extend(Actor5)
        
        # # Get Budgetbin
        if request.form["budgetbin"]:
            print(request.form["budgetbin"])
            Budgetbin = request.form["budgetbin"]
            result['budgetbin'][str(Budgetbin)] = 1
        
        # convert the disctinary back to a dataframe
        res = pd.DataFrame(result)

        # remove null features
        feat = []
        inpt =[]
        for col in res.columns:
            feat.extend(res[col].dropna().index)
            inpt.extend(res[col].dropna().values)

        # Random Forest prediction
        rf_prediction = rf_model.predict([inpt])

        if rf_prediction[0] == 0:
            rfresult = 'Random Forest Prediction: Box office failure'
        
        if rf_prediction[0] == 1:
            rfresult = 'Random Forest Prediction: Box office sucess'

        # # reshape data for Deep Learning model input
        # inptexpanded = np.expand_dims(inpt, axis=0)
        # print(inptexpanded)

        # # Deep Learning model prediction
        # # nn_prediction = nn_model.predict_classes(inptexpanded)
        # nn_prediction = nn_model._make_predict_function(inptexpanded)
        
        # print(nn_prediction)

        # # populated model result
        # if nn_prediction[0] == 0:
        #     nnresult = 'Neural Network Prediction: Box office failure'
        
        # if nn_prediction[0] == 1:
        #     nnresult = 'Neural Network Prediction: Box office success'
        
        # Logistic Regression prediction
        lr_prediction = lr_model.predict([inpt])

        if lr_prediction[0] == 0:
            lrresult = 'Logistic Regression Prediction: Box office failure'
        
        if lr_prediction[0] == 1:
            lrresult = 'Logistic Regression Prediction: Box office success'

        # print(f'Random Forest result: {rf_prediction}.')
        # print(f'Logistic Regression result: {result}.')
        # print(f'Neural Network result: {nn_prediction}.')
        
        # Add input and results to MondgoDB and redirect to predictionresults page to display results
        return render_template("predictionresult.html",lrresult = lrresult, rfresult = rfresult)
        # return render_template("predictionresult.html",lrresult = lrresult, rfresult = rfresult, nnresult=nnresult)
    
    return render_template('MovieG.html')

@app.route('/process')
def process():
    return render_template('process.html')

@app.route('/fun')
def fun():
    return render_template('Fun.html')

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/pie')
def pie():
    return render_template('pie_viz.html')

@app.route('/chart3D')
def chart3D():
    return render_template('3D ROI.html')

@app.route('/documentary')
def documentary():
    return render_template('documentary_viz.html')

@app.route('/genreROI')
def genreROI():
    return render_template('genreROI_viz.html')

@app.route('/adventure')
def adventure():
    return render_template('adventure_viz.html')

@app.route('/genrePopularity')
def genrePopularity():
    return render_template('genrePopularity_viz.html')

# Route for acknowledgments
@app.route('/acknowledgments')
def acknowledgments():
    return render_template('acknowledgments.html')

# Route for directors
@app.route('/directors',methods=['GET', 'POST'])
def directors():
    
    # Find one record of data from the mongo database
    output=[]
    for s in mongo.db.directors_distinct.find():
        output.append(s['crew_name'])

    return jsonify(output)

# Route that will trigger the scrape function

# Route for actors
@app.route('/actors',methods=['GET', 'POST'])
def actors():
    # import pandas as pd
    # Find one record of data from the mongo database
    output=[]
    for s in mongo.db.actors_distinct.find():
        output.append(s['actor_name'])

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
