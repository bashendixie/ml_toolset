from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

week_day_dict = {
    0 : 'Streptococcus_pyogenes',
    1 : 'Salmonella_enterica',
    2 : 'Enterococcus_hirae',
    3 : 'Escherichia_coli',
    4 : 'Campylobacter_jejuni',
    5 : 'Streptococcus_pneumoniae',
    6 : 'Staphylococcus_aureus',
    7 : 'Escherichia_fergusonii',
    8 : 'Bacteroides_fragilis',
    9 : 'Klebsiella_pneumoniae',
}

f2=open('svm_v3.model','rb')
s2=f2.read()
model1=pickle.loads(s2)

data = pd.read_csv('data/test.csv')
data = (data - data.min()) / (data.max() - data.min())
test_X = data.to_numpy()

predictions = model1.predict(test_X)

preds = []
for pred in predictions:
    preds.append(week_day_dict[pred])

res = pd.DataFrame()
res['target'] = preds
res.to_csv("predict_svm_v3.csv")
