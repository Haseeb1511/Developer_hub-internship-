import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt



with open("hier.pkl","rb") as f:
    agg = pickle.load(f)
with open("kmeans.pkl","rb") as f:
    kmeans = pickle.load(f)

st.title("Customer Segmentation")

choice = st.sidebar.radio("Chose a model",("KMeans","AgglomerativeClustering"))
if choice=="KMeans":
    select_model = kmeans
else:
    select_model=agg

annaul_income = st.number_input("Annual Income(K$)",0,140,20)
spending_score = st.number_input("Spending score",0,100,30)


st.text("Chose the Annual income and Spending score")
if st.button("Check Cluster"):
    new_data = np.array([[annaul_income,spending_score]])
    pred = select_model.predict(new_data)[0]
    st.success(f"The given customer belong to cluster : {pred}")
    if choice=="KMeans":
        st.image("customer_plot.png")
    else:
        st.image("dendogram.png")



