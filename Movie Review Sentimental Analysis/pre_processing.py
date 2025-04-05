import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

nltk.download("stopwords")
nltk.download("wordnet")

wnl = WordNetLemmatizer()
sw= set(stopwords.words("english"))    

def pre_processing(text):
    text = re.sub(r"https?://\S+"," ",text)
    text = re.sub(r'[^a-zA-Z\s]', " ", text)
    text = text.lower().split()
    text = " ".join(wnl.lemmatize(word) for word in text if word not in sw )
    return text

def main(text_series):
    if isinstance(text_series,list):
        text_series = pd.Series(text_series)
    return text_series.astype(str).apply(pre_processing)

def dense(data):
    return data.toarray()