import streamlit as st
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model

st.title("Customer Churn Detection")
st.write("""
Created by Galih S
""")

# import model

pipe_prep = pickle.load(open("pipe_prep.pkl", "rb"))
model = load_model('model_last/')



st.write('Insert feature below to predict')

# user input
region = st.radio("Region",('Town', 'City', 'Village'))
membership = st.selectbox(label='Membership', options=['Basic Membership','No Membership','Gold Membership','Silver Membership','Premium Membership','Platinum Membership'])
referral = st.radio("With Referral",('No','Yes'))
medium =  st.radio("Operation Media",('Smartphone','Desktop','Both'))
avg_time_spent = st.number_input(label='Time spent on website (in minute)', min_value=0, max_value=3235, value=100, step=1)
avg_trans = st.number_input(label='Transaction value', min_value=800, max_value=99990, value=1000, step=1)
frequent_login = st.number_input(label='Login on website frequence', min_value=0, max_value=73, value=1, step=1)
points = st.number_input(label='Points in wallet', min_value=0, max_value=2069, value=100, step=1)

st.write('Give your feedback to maintain our services')
feedback = st.selectbox(label='Customer Feedback', options=['Too many ads', 'No reason specified', 'Reasonable Price','Quality Customer Care', 'Poor Website', 'Poor Customer Service','Poor Product Quality', 'User Friendly Website','Products always in Stock'])


# convert into dataframe
data = pd.DataFrame({'region_category': [region],
                'membership_category': [membership],
                'joined_through_referral': [referral],
                'medium_of_operation':[medium],
                'avg_time_spent': [avg_time_spent],
                'avg_transaction_value': [avg_trans],
                'avg_frequency_login_days': [frequent_login],
                'points_in_wallet': [points],
                'feedback': [feedback],
                })

# model predict
if st.button('Predict'):
    prep = pipe_prep.transform(data)
    pred = model.predict(prep)
    pred2 = np.where(pred >= 0.5, 1, 0)


    if pred2 == 1:
        pred2 = 'Churn Detected'
    else:
        pred2 = 'None'

    st.write('Based on the features, the result is: ')
    st.write(pred2)
