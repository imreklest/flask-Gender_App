from flask import Flask, redirect, request, render_template, session
from flask_session import Session
from keras.models import load_model
from tensorflow.keras.utils import img_to_array # type: ignore
from werkzeug.utils import secure_filename
import pandas as pd
import cv2
import numpy as np
import os

app = Flask(__name__)

Session(app)

# Process image and predict label
def processImg(IMG_PATH):
    # Read image
    model = load_model("saved_models/bestModel.h5")
    
    # Preprocess image
    image = cv2.imread(IMG_PATH)
    image = cv2.resize(image, (75, 75))

    # Apply histogram equalization for contrast enhancement
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(gray_image)
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

    # Convert image to array
    image = img_to_array(enhanced_image)
    image = np.expand_dims(image, axis=0)

    res = model.predict(image, batch_size=32)
    
    if res[0][0] > 0.5:
        gender = "Male"
    else:
        gender = "Female"

    return gender

#tes upload file
ALLOWED_EXTENSION = set(['png','jpg','jpeg'])
app.config['UPLOAD_FOLDER'] ='static/img/'

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSION

@app.route("/")
def index():
	return render_template('home.html')

@app.route("/home")
def home():
	return render_template('index.html')

@app.route("/accuracy")
def accuracy():
	return render_template('accuracy.html')

@app.route("/about")
def about():
	return render_template('about.html')

@app.route("/dataset")
def dataset():
	return render_template('dataset.html')

@app.route("/result", methods=['POST'])
def result():
	if request.method == 'POST':
		file = request.files['file']
		if 'file' not in request.files:
			return redirect(request.url)
		if file.filename == '':
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'] + filename))
			foto = os.path.join(app.config['UPLOAD_FOLDER'] + filename)
			resp = processImg(os.path.join(app.config['UPLOAD_FOLDER'] + filename))
			return render_template('hasil.html', hasil=resp, foto=foto)

if __name__ == '__main__':
	app.run(debug = True)