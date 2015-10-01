import csv
import nltk
import string

from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import linear_kernel
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

with open('Data/training_3.csv', 'rb') as train:
    reader = csv.reader(train)
    f = []
    filtered = []
    for row in reader:
        essay = row[2].translate(None, string.punctuation).decode('utf-8')
        tokens = tokenize(essay)
        filtered.extend([w for w in tokens if not w in stopwords.words('english')])

count = Counter(filtered)
d = [x[0] for x in count.most_common(100)]
print d
f = [k for k in filtered if k in d]
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english').fit_transform(f)
cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
related = cosine_similarities.argsort()[:-5:-1]


