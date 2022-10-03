import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__,static_url_path='/static')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')


@app.route('/about',methods=['GET','POST'])
def about():
    return render_template('about.html')


@app.route('/values',methods=['POST'])
def values():
    return render_template('predict.html')


@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()] #since all input values are of float type, it was more convenient to store them in a list that contains only float values
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 0:
        return render_template('bad_wine.html')
    elif output == 1:
        return render_template('good_wine.html')



if __name__ == "__main__":
    app.run(debug=True)