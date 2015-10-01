import csv
import nltk
import string

from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(f):
    tokens = nltk.word_tokenize(f)
    stems = stem_tokens(tokens, stemmer)
    return stems

with open('training_3.csv', 'rb') as train:
    reader = csv.reader(train)
    f = []
    for row in reader:
        essay = row[2].decode('utf-8')

        f.append(row[2].translate(None, string.punctuation).decode('utf-8'))

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(f)
feature_names = tfidf.get_feature_names()
response = tfidf.transform(f)
for col in response.nonzero()[1]:
        print feature_names[col], ' - ', response[0, col]
