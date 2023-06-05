# -*- coding: utf-8 -*-
"""


"""


import pandas as pd
import streamlit as st 
from pickle import dump
from pickle import load

st.title('Model Deployment: Twitter Semantic Analysis')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Tweets = st.sidebar.selectbox('figurative',('1','0'))
    Class = st.sidebar.selectbox('irony',('1','0'))
    data = {'Tweets':Tweets,
            'Class':Class}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('C:\Users\Pooja Patil\Downloads\tweet.csv', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)


