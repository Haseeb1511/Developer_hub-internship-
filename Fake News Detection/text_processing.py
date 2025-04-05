import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Download stopwords and wordnet
nltk.download('stopwords')
nltk.download('wordnet')


def to_dense(x):
    return x.toarray()

def convert(text_series):
    # Ensure input is a Pandas Series as pipeline take panda series while the below code run row by row
    if isinstance(text_series, list):
        text_series = pd.Series(text_series)

    return text_series.astype(str).apply(process_text)

def process_text(text):
    wn = WordNetLemmatizer()
    stopword = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z"]', " ", text) 
    text = text.lower()
    text = " ".join(wn.lemmatize(word) for word in text.split() if word not in stopword)
    return text

