import pickle 
from flask import Flask
import numpy as np
with open('/content/sample_data/wine_rf.pkl') as model_file:
  model=pickle.load(model_file)

app=Flask(__name__)
@app.route('/predict')
def predict_wine():
    f1= request.args.get("F1")
    f2= request.args.get("F2")
    f3= request.args.get("F3")
    f4= request.args.get("F4")
    f5= request.args.get("F5")
    f6= request.args.get("F6")
    f7= request.args.get("F7")
    f8= request.args.get("F8")
    f9= request.args.get("F9")
    f10= request.args.get("F10")
    f11= request.args.get("F11")
    f12= request.args.get("F12")
    f13= request.args.get("F13")
    prediction = model.predict(np.array([[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13]]))
    return str(prediction)
if __name__=='__main__':
  app.run()
