import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
import os
from config import *
###########################################################################################################################

st.set_page_config(
page_title = 'AOML Project',
page_icon = ':memo:',
layout = 'wide',
)
###########################################################################################################################

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked=False    

if 'options' not in st.session_state:
    st.session_state['options'] = ""

def callback():
    st.session_state.button_clicked=True
###########################################################################################################################

maps={
    1:'model_bi_grams',
    2:'model_tri_grams',
    3:'model_four_grams',
    4:'model_five_grams',
}
###########################################################################################################################

with open('tokenizer.pickle', 'rb') as handle:
  tokenizer=pickle.load(handle)
model_dict=get_models()
###########################################################################################################################

st.markdown("<h1 style='text-align: center; color: white;'>Next Word Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>It's elementary my dear ______</h3>", unsafe_allow_html=True)

placeholder=st.empty()
text=placeholder.text_input('Enter the text: ', value=st.session_state['options'])

length=len(text.split())

if length>0:

    n=length if length<=3 else 4
    st.text(f'Since length={length}, choosing a {maps[n][6:]} model')

    sequence=get_sequences(tokenizer, text, n)
    try:
    # st.write(sequence)
        preds=get_predictions(model_dict[maps[n]], np.array(sequence), length)
        
        # predicted_word={}
        st.text('Words you may require...')
        for i in preds:
            key=list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(i)]
            if st.button(key):
                st.session_state['options']=f'{text} {key}'
                   
    except Exception as e:
        # st.write(e)
        st.text('No Further Predictions Available...\nTry some other words.')