import streamlit as st
from pre_processing import main,dense
import pickle

with open("pipe.pkl","rb") as f:
    model = pickle.load(f)


st.title("Movie Review Sentiment Analysis")

st.text("Enter review below")

user_input = st.text_area("type here")

if st.button("Check"):
    if user_input.strip():
        pred = model.predict([user_input])[0]
        if pred==1:
            st.success("Positive Sentiment") 
            st.balloons()
        else:
            st.error("Negative Sentiment")
else:
        st.warning("⚠️ Please enter a review before checking sentiment.")

#the movie was very bad and stoy was poor

import streamlit as st

