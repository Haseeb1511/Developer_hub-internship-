import streamlit as st
import pickle
from text_processing import convert,to_dense,process_text

with open("gnb_pipeline.pkl","rb") as f:
    gnb_pipe = pickle.load(f)

with open("rf_pipeline.pkl","rb") as f:
    rf_pipe = pickle.load(f)

st.sidebar.header("Chose Model")
choice = st.sidebar.radio("make choice",("GaussianNB","Random Forest"))
if choice =="GaussianNb":
    select_model = gnb_pipe
else:
    select_model = rf_pipe



st.title("Fake News Detection")

st.text("Enter aurthor name and title of newspaper below")

user_input = st.text_area("enter text")

if st.button("Predict"):
    if user_input.strip():
        pred = select_model.predict([user_input])[0]
        result = "FAKE NEWS" if pred==1 else " REAL NEWS"
        if result==" REAL NEWS":
            st.balloons()
        st.write(f"prediction is {result}")

else:
    st.warning("Please enter some text before pressing predict")

st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: white;
        }
    </style>
    <div class="footer">
        Made with ❤️ using Streamlit | © 2025 Haseeb manzoor
    </div>
""", unsafe_allow_html=True)

