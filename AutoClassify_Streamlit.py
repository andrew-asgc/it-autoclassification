#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import streamlit as st

# load the model from disks
vectorizer = pickle.load(open("./vector.pickel", "rb"))
model = pickle.load(open('./it_auto_classification.sav', 'rb'))
identify = pd.read_csv('./identify.csv')

st.write('''# IT Expenditure Auto Classification''')



option = st.selectbox(
'How would you like to enter the data',
('Individual Input', 'Batch Input'))

if option=='Individual Input':
	st.write('''### Enter Item Description''')
	new_item = st.text_input('Type below:')
	print ('\nMost likely Sub Class: ', model.predict(vectorizer.transform([new_item])))
	identify['Probability (%)'] = model.predict_proba(vectorizer.transform([new_item]))[0]*100
	df = identify.sort_values(by='Probability (%)', ascending=False)
	df.columns = [['Line Type','UNSPSC Class Name','Probability (%)']]

	st.write('''### The most likely UNSPSC Class for the above item is '''+'**'+df.iloc[0,1]+'**'+' with '+'**'+str(df.iloc[0,-1])+'**'+'% confidence.')
	st.write('''### The following are the top 5 most likely UNSPSC Classes''')
	st.write(df.head().reset_index(drop=True))

else:
	st.write('''### Upload Excel File''')
	file = st.file_uploader("Choose an excel file", type="xlsx")
	st.write(pd.read_excel(file))
