
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
import sys
import pickle

# This will only run for names with one first name and one last name e.g. "Narendra Modi"
def extract_features(data):
    test_data = {}
    if len(data) == 2:
        test_data['first_name'] = data[0]
        test_data['last_name'] = data[1]
    elif len(data) == 1:
        test_data['first_name'] = data[0]
        test_data['last_name'] = np.NaN
    
    test_data['f1'] = test_data['first_name'][:1]
    test_data['f2'] = test_data['first_name'][0:2]
    test_data['f3'] = test_data['first_name'][0:3]
    test_data['l1'] = test_data['last_name'][-1]
    test_data['l2'] = test_data['last_name'][-2:]
    test_data['l3'] = test_data['last_name'][-3:]
    test_data['flen'] = len(test_data['first_name'])
    test_data['llen'] = len(test_data['last_name'])
    return test_data

def load_model(filename):
    pickle_loaded = pickle.load(open(filename, 'rb'))
    loaded_model = pickle_loaded[0]
    loaded_vectorizer = pickle_loaded[1]
    return loaded_model, loaded_vectorizer


# Pass input via command line in quotes
data = sys.argv[1]
data = data.lower()
data = data.split(' ')
print(data)
if '' in data:
    data.remove('')
if "\n" in data:
    data.remove("\n")
print(data)
test_data = extract_features(data)
print(test_data)
model, vectorizer = load_model('Surabhi_Singhal_spotDraft_model.sav')
print(type(model))

print("Prediction", model.predict(vectorizer.transform(test_data)))

