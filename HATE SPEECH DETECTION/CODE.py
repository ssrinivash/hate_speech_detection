import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

#   using twitter.csv file
data = pd.read_csv("twitter.csv")
print(data.sample(5))

data = data[["CONTENT", "CLASS"]]
print(data.sample(5))

data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1: "Spam"})
print(data.sample(5))

x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])

cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = BernoulliNB()
model.fit(xtrain, ytrain)

# Model accuracy
y_pred = model.predict(xtest)
accuracy = accuracy_score(ytest, y_pred)
print("YouTube Spam Detection Accuracy:", accuracy)

sample = "i think about 100 millions of the views come from people who only wanted to check the views"
data = cv.transform([sample]).toarray()

print("Prediction for sample:", model.predict(data))


# Hate Speech Detection (now using twitter.csv)
import re
import nltk
from sklearn.tree import DecisionTreeClassifier

stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string

stopword = set(stopwords.words('english'))
data = pd.read_csv("twitter.csv")
print(data.head())

data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
print(data.head())

data = data[["tweet", "labels"]]
print(data.head())

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

data["tweet"] = data["tweet"].apply(clean)

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x)  # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Model accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Hate Speech Detection Accuracy:", accuracy)

sample = "good kill"
data = cv.transform([sample]).toarray()

print("Prediction for sample:", clf.predict(data))
