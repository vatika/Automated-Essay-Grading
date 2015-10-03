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
c = []
with open('Data/training_3.csv', 'rb') as train:
    reader = csv.reader(train)
    all_words = []
    essays = []
    for row in reader:
        essay = row[2].translate(None, string.punctuation).decode('utf-8')
        tokens = tokenize(essay)
        all_words.extend(tokens)
        c.append(tokens)
        essays.append(essay)

count = Counter(all_words)

#most common words in all essays
common_words = [x[0] for x in count.most_common(10)]

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(essays)

feature_names = tfidf.get_feature_names()
v = []
for i in xrange(len(c)):
   value = 0
   c_e = Counter(c[i])
   common_10_e = c_e.most_common(10)
   common_words_e = [x[0] for x in c_e.most_common(10)]
   for j in xrange(len(common_words_e)):
       if common_words_e[j] in common_words:
           for col in tfs.nonzero()[1]:
              if feature_names[col] == common_words_e[j]:
                       value += common_10_e[j][1]*tfs[0,col]
                       break
   v.append(value)
print v
#cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
#related = cosine_similarities.argsort()[:-5:-1]

