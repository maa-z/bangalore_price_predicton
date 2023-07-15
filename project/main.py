from flask import Flask , render_template , request
import pandas as pd
import pickle
import numpy as np



app = Flask(__name__)

df = pd.read_csv("clean_data.csv")
# print(df.head())

pipe = pickle.load(open("RidgeModel.pkl","rb"))

locations = sorted(df["location"].unique())

@app.route('/')
def index():
    return render_template('index.html',locations=locations)


@app.route('/predict',methods=['POST'])
def predict():

    location = "Anandapura"
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))
    print(bhk)
    print(location)
    print(sqft)
    print(bath)

    input = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction = np.round(pipe.predict(input)[0]*(-1*100000),2)
    print(prediction)
    return str(prediction)



    return ""


if __name__=='__main__':
    app.run(debug=True,port=5000) 