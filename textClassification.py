import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import nltk
from sklearn import model_selection


df = pd.read_table('text_docs/SMSSpamCollection', header=None, encoding='utf-8')

print(df.head())

classes = df[0]
print(classes.value_counts())

encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

print(Y[:10])

text_messages = df[1]
print(text_messages[:10])

#change email address to emailaddress
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                      'emailaddress')

#change url to webaddress
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')

# change money symbol
processed = processed.str.replace(r'£|\$', 'moneysymb')

# replace phone number
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')

# number to numbr
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.lower()
print(processed)



stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

all_words = []

for message in processed:
    words = nltk.word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)

print('no. of words',len(all_words))
print('most common words',all_words.most_common(15))

word_features = list(all_words.keys())[:1500]

def find_features(message):
    words = nltk.word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

features = find_features(processed[0])
for key, value in features.items():
    if value == True:
        print(key)


messages = list(zip(processed, Y))

np.random.shuffle(messages)

featuresets = [(find_features(text), label) for (text, label) in messages]
from sklearn import model_selection
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=1)

print(len(training))
print(len(testing))

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
model = SklearnClassifier(SVC(kernel = 'linear'))
model.train(training)

accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy",accuracy)

# use many algorithm in single program like this
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))
