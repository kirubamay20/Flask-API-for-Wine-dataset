import pickle 
from flask import Flask
import pandas as pd

import numpy as np
with open('/content/sample_data/wine_rf.pkl', 'rb') as model_file:
  model=pickle.load(model_file)

app=Flask(__name__)
@app.route('/predict_file', methods=['POST'])
def predict_wine_file():
    input_data = pd.read_csv(request.files.get["input_file"])
    prediction = model.predict(input_data)
    return str(list(prediction))
if __name__=='__main__':
  app.run()
