# inport library
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer,util
from numpy.linalg import norm
import pickle

dataset = pd.read_excel('./model train files/ChankyaNeeti_Description.xlsx')
c_des = list(dataset['Description'])
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

trained_des = model.encode(c_des)

data = {'dataset': dataset,
        'c_des': c_des,
        'trained_des' : trained_des,
        'model': model,
        }

with open('model.pickle', 'wb') as f:
  pickle.dump(data, f)