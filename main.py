import pandas as pd
import numpy as np

df_train = pd.read_csv('MOVIE GENRE CLASSIFICATION\\Data\\train_data.txt', sep = ":::", names = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine = 'python')
x = df_train['DESCRIPTION'][:5000]
y = df_train['GENRE'][:5000]

# Preprocessing & Bag of words
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    words = text.lower().split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    words = ' '.join(words)
    return words
x = [preprocess_text(doc) for doc in x]

# TFIDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features = 5000)
x = vectorizer.fit_transform(x)

# Labelencoding y
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Predict genre
def predict_genre (description, model):
    des = preprocess_text(description)
    des = vectorizer.transform([des])
    predicted_genre = model.predict(des)
    genre_name = encoder.inverse_transform(predicted_genre)[0]
    return genre_name

# Model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

list = [LogisticRegression(), MultinomialNB(), SVC()]
description = 'His father has died, he hasnt spoken with his brother for about 10 years and has a serious cancer. Diego is a talented film director with difficulty to deal with his sickness, which is making him lose his friends and family. His best friend and doctor Ricardo gives him the news that he needs a bone marrow transplantation, otherwise hell die. He gets married to a beautiful woman, Livia, just before going to Seattle to get treatment. There, he undergoes numerous medical procedures. During treatment, he meets an Hindu boy, with whom he plays and whom he tells amazing stories. Odds are against him and when stakes are the highest, Diego gets a visit from a very uncommon man.'
for i in list:
    i.fit(x_train, y_train)
    y_pred = i.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    predicted_genre = predict_genre(description, i)
    print(f"Predicted Genre for above description using {i} is: {predicted_genre}")
    print(f"Accuracy for {i}: {accuracy: .2f}")