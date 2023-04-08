import tensorflow as tf
import numpy as np
import os
import streamlit as st
import gdown

@st.experimental_singleton
def get_models():
    if not os.path.exists('models'):
        gdown.download_folder('https://drive.google.com/drive/folders/1vDuO3_3zlViFSbndcjHhIgjf2z4QNbV1?usp=sharing', quiet=True)
    models={}
    for i in os.listdir('models'):
        models[i.split('.')[0]]=tf.keras.models.load_model(f'models/{i}', compile=False)
        models[i.split('.')[0]].compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    return models

def get_sequences(tokenizer, text, n):
    return tokenizer.texts_to_sequences([text.lower().split()[-n:]])[0]

def get_predictions(model, sequence, length):
    print(model)
    if length>1:
        sequence=sequence[np.newaxis,]

    return np.flip(np.argsort(model.predict(sequence, verbose=0), axis=1)[0][-10:])
