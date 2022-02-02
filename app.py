from flask import Flask, request, render_template
from flask_cors import cross_origin
import PIL
import sklearn
import tensorflow
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import PIL

import pandas as pd

app = Flask(__name__)
#model = pickle.load(open("model_pickle", "rb"))
model = load_model("model1.h5")

def predict_label(path):
    test_image = image.load_img(path, target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    
    result = model.predict(test_image)
    
    return result







@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")



@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        img = request.files["file"]
        img_path = "static/" + img.filename
        img.save(img_path)
        #print(img)

        p = predict_label(img_path)
        
        if p[0][0] < 0.5:
            output =  'Normal'
        elif p[0][0] >= 0.5:
            output =  'Pneumonia'

        if output == 'Pneumonia':
            return render_template('home.html',prediction_text=" There is high chance of you having a {}".format(output))
        else:
            return render_template('home.html',prediction_text=" From your X-ray image we can see your heart is  {}".format(output))

    




        
        #return render_template('home.html',prediction_text=" Chance of you having a Pneumonia is {}".format(output))

    return render_template("home.html")









if __name__ == "__main__":
    app.run(debug=True)