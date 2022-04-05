import pickle
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)
import sklearn
print(sklearn.__version__)
# Load the model
model = pickle.load(open('pickle_dump_model.pkl', 'rb'))

def text_to_ndarray(text):
    return np.array([float(x) for x in text.split(',')])
@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    abc = text_to_ndarray(data['exp'])

    prediction = model.predict([abc])
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(str(output))
if __name__ == '__main__':
    app.run(port=5000, debug=True)


