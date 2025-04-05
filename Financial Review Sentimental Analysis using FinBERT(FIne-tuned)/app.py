import streamlit as st
from pre_processing import pre_processing
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


st.title("Financial Sentiment Analysis")

fine_tuned_model_path = "./model"
fine_tuned_tokenizer_path = "./tokenizer"

model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_tokenizer_path)

st.text("Enter text below")
user_input = st.text_area("Enter text here")
if st.button("Predict"):
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    cleaned = pre_processing(user_input)
    result = pipe(cleaned)[0]
    st.success(result["label"])


