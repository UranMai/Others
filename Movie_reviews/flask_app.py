from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
#from sklearn.externals import joblib
import joblib
loaded_model=joblib.load(open("pkl_objects/model.pkl", 'rb'))
loaded_stop=joblib.load(open("./pkl_objects/stopwords.pkl", 'rb'))
loaded_vec=joblib.load(open("./pkl_objects/vectorizer.pkl", 'rb'))
model1=joblib.load(open("./pkl_objects/model_ranking", 'rb'))
# vec1 = joblib.load(open("./pkl_objects/vec1.pkl", 'rb'))

app = Flask(__name__)

def sentiment(document):
    label = {0: 'negative', 1: 'positive'}
    X = loaded_vec.transform([document])
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))

    # X1=vec1.transform([document])
    ranking = model1.predict([document])[0]
    return label[y], proba, ranking

def ranking(document):
    X = vec.transform([document])
    y = model1.predict(X)[0]
    return y

class ReviewForm(Form):
    moviereview = TextAreaField('',[validators.DataRequired(), validators.length(min=15)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba, ranking = sentiment(review)
        #y = ranking(review)
        return render_template('results.html',content=review,prediction=y,probability=round(proba*100, 2), rank=ranking)
    return render_template('reviewform.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
