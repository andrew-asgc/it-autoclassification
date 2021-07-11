#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import streamlit as st

# function to download
import base64
def get_table_download_link(df):
	"""Generates a link allowing the data in a given panda dataframe to be downloaded
	in:  dataframe
	out: href string
	"""
	csv = df.to_csv(index=False)
	b64 = base64.b64encode(
		csv.encode()
	).decode()  # some strings <-> bytes conversions necessary here
	return f'<a href="data:file/csv;base64,{b64}" download="file.csv">Download csv file</a>'

# load the model from disks		
vectorizer = pickle.load(open("./vector.pickel", "rb"))
model = pickle.load(open('./it_auto_classification.sav', 'rb'))
identify = pd.read_csv('./identify.csv')
st.markdown('<style>.reportview-container{background: url(https://github.com/andrew-asgc/it-autoclassification/blob/main/Picture1.png?raw=true); background-size: cover;}</style>',unsafe_allow_html=True)

st.title('''IT Expenditure Auto Classification''')

option = st.sidebar.radio(
    "Input Method:",
    ('Individual Input','Batch Input'))

if option=='Individual Input':
	st.write('''### Enter Item Description''')
	new_item = st.text_input('Type below:')
	if new_item=='':
		st.write('')
	else:	
		identify['Probability (%)'] = model.predict_proba(vectorizer.transform([new_item]))[0]*100
		df = identify.sort_values(by='Probability (%)', ascending=False)

		st.write('The most likely UNSPSC Class for the above item is **{0}** with **{1:.1f}**% confidence.'.format(df.iloc[0,1], df.iloc[0,-1]))
		st.empty()
		with st.beta_expander('See other options'):
			st.write('''### The following are the top 5 most likely UNSPSC Classes''')
			st.table(df.iloc[:,1:].head().reset_index(drop=True))
else:
	st.write('''
		To Auto-Classify a batch, please upload an Excel file with each Item Description listed in **_1 column only_**.
		''')
	st.write('''### Upload Excel File''')
	file = st.file_uploader("Choose an excel file", type="xlsx")
	if file==None:
		st.write('')
	else:
		read_file = pd.read_excel(file, engine='openpyxl', header=0)
		read_file = pd.DataFrame(read_file.iloc[:,1])
		read_file.columns=['Item Description']
   	 
		read_file['Sub Sub Class'] = model.predict(vectorizer.transform(read_file.iloc[:,0].values.flatten()))
		read_file['Probability (%)'] = model.predict_proba(vectorizer.transform(read_file.iloc[:,0].values.flatten())).max(1)*100
		read_file = read_file.merge(identify, how='left', left_on='Sub Sub Class', right_on='sub_sub_class')
		read_file = read_file[['item_code','Item Description', 'main_class', 'sub_class', 'sub_sub_class']]

		st.write('The following are the most likely UNSPSC Class Names for the uploaded Item Descriptions:')
				
		# download link
		st.markdown(get_table_download_link(read_file), unsafe_allow_html=True)
		st.table(read_file)

