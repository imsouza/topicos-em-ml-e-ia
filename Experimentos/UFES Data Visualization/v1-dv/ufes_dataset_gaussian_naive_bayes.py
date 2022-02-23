#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import pandas as pd
import numpy as np
import matplotlib
import warnings
import csv

sns.set(style="white", color_codes=True)
warnings.filterwarnings("ignore")

url = 'http://dados.ufes.br/dataset/37ba75f0-333d-475a-b60a-8b87ac7e2129/resource/2de89c23-2580-4fff-a02b-e82642970dcd/download/alunos.csv'
raw_data = urllib.request.urlopen(url)
names=['ALUNOS', 'CURSO', 'PERÍODO DE INSCRIÇÃO', 'ANO DE INSCRIÇÃO']

dataset = pd.read_csv(raw_data, header=None, sep=',', names=names)

dataset.head()

df = pd.DataFrame(data=dataset, columns=names)
df = df.iloc[1:]
df

features = ['ALUNOS', 'PERÍODO DE INSCRIÇÃO', 'ANO DE INSCRIÇÃO']
labels = ['CURSO']

label_encoder = preprocessing.LabelEncoder() 

df['ALUNOS'] = label_encoder.fit_transform(df['ALUNOS']) 
df['PERÍODO DE INSCRIÇÃO'] = label_encoder.fit_transform(df['PERÍODO DE INSCRIÇÃO'])
df['ANO DE INSCRIÇÃO'] = label_encoder.fit_transform(df['ANO DE INSCRIÇÃO']) 

X = df[features]
y = df[labels]

features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.43, random_state=42)

df['CURSO'].value_counts()

clf = GaussianNB()

model = clf.fit(features_train, labels_train)

pred = model.predict(features_test)

get_ipython().run_line_magic('matplotlib', 'inline')

diagram = sns.pairplot(df.drop("ANO DE INSCRIÇÃO", axis=1), hue="CURSO", size=6, diag_kind="kde")

plt.savefig('output.png', bbox_inches='tight', dpi=300)
