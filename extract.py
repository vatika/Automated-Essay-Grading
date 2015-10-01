import csv
import nltk
import string

from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('corpora')
nltk.download('punkt')
def get_tokens(f):
    t = []
    for i in f:
            lowers = i.lower()
            no_punctuation = lowers.translate(None, string.punctuation)
            tokens = nltk.word_tokenize(no_punctuation)
            t.append(tokens)
    return t

with open('training_set_rel3.csv', 'rb') as train:
    reader = csv.reader(train)
    f = []
    for row in reader:
        f.append(row[2])

tokens = get_tokens(f)
filtered = [w for w in tokens if not w in stopwords.words('english')]
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(tokens)
print tfs

