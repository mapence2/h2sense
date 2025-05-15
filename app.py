from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle  # or use pickle
import os

app = Flask(__name__)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_rows = data['inputs']

    X = np.array([[row['temp'], row['response'], row['humidity']] for row in input_rows])
    means, stds = model.predict(X,return_std=True)

    return jsonify(predictions=[{'mean': float(m), 'std': float(s)} for m, s in zip(means, stds)])
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)