import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

sw = set(stopwords.words("english"))
wn = WordNetLemmatizer()

def pre_processing(text):
  text = text.translate(str.maketrans("","",string.punctuation))
  text = text.lower().split()
  text = " ".join(wn.lemmatize(word) for word in text if word not in sw)
  return text