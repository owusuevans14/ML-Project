import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('model')
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    final_features = [[[int(x)] for x in request.form.values()]]
    prediction = model.predict(final_features)
    
    output = round(prediction[0][0], 2)

    return render_template('index.html', prediction_text='TOTAL CASE COUNT WOULD BE {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([[[i] for i in list(data.values())]])
    #return {'pplea':'hihi'}
    output = prediction[0][0]
    return str(output)

if __name__ == "__main__":
    app.run(debug=True)