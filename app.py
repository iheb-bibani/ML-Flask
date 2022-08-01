import numpy as np

from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.sav')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text= 'MSFT Stock Price is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True,port=5002)