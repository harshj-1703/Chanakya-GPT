from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sentence_transformers import util

# initializing flask
app = Flask('GeetaGPT')

with open('model.pickle', 'rb') as f:
    data = pickle.load(f)

dataset = data['dataset']
c_des = data['c_des']
model = data['model']
cosine_similarities = []
trained_des = data['trained_des']

@app.route('/',methods=['POST','GET'])
def results():
    cosine_similarities = []
    if request.args.get('description') != '':
        
        description = request.args.get('description')
        
        new_description = model.encode(description)
        cosine_similarities = util.dot_score(new_description, trained_des)
        index = (np.argmax(cosine_similarities)).item()
        # print(index)
        output1 = dataset['Description'][index]
        output_chapter_no = dataset['Chapter'][index]
        output_slok_no = dataset['No'][index]
        return render_template('index.html', description=description, output_des=output1, slok=output_slok_no, chapter=output_chapter_no)
    else:
        return render_template('index.html')


# run file
app.run("localhost", "9999", debug=True)