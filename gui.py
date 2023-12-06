import os
import cv2
from path import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from flask import *
from werkzeug.utils import secure_filename
from PIL import Image
from joblib import dump, load

app= Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def message():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        run_model(filename)
        f=open("filename.txt", "r")
        content = f.read()
        print(content)
        return render_template('index.html', filename=os.path.join(app.config['UPLOAD_FOLDER'], filename),text=content)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

    
def run_model(filename):
    img = cv2.imread('./static/uploads/' + filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow('img', img_gray)
    cv2.waitKey(2000)
    '''2 Binary the grayscale image'''
    ret, img_threshold = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('img', img_threshold)
    cv2.waitKey(2000)
    img_final = np.asarray(img_threshold)
    img=np.asarray(img_final)
    img_final = cv2.resize(img_threshold, (28, 28))
    cv2.imshow('final1', img_final)
    cv2.waitKey(2000)
    print(img_final.shape)
    img_final = np.reshape(img_final, (784,))
    print(img_final.shape)
    img_final=img_final.reshape(-1, 784)

    f = open("filename.txt", "w")

    #.................prediction...................
    svm = load('svm.joblib')
    pred = svm.predict(img_final)
    s = 'SVM - ' + str(pred)[1] + '\n'
    f.write(s)

    print('decision tree')
    dectree = load('decision_tree.joblib')
    pred = dectree.predict(img_final)
    print(pred)
    s='Decision Tree - ' + str(pred)[1] + '\n'
    f.write(s)

    logireg = load('logistic_regression.joblib')
    pred = logireg.predict(img_final)
    s = 'Logistic Regression - ' + str(pred)[1] + '\n'
    f.write(s)
    f.close()



if __name__ == '__main__':
    app.run(debug=True)
    # run_model('d_3_h.png')

