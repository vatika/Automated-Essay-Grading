# Copyright 2015 - Anurag Ghosh, Vatika Harlalka, Abhijeet Kumar
import csv
import nltk
import string

from collections import Counter

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

import enchant

enchant_dict = enchant.Dict("en_US")
stemmer = PorterStemmer()

class Point:
    def __init__(self,essay_id,essay_set,essay_str,score):
        self.essay_id = essay_id
        self.essay_set = essay_set
        self.essay_str = essay_str
        self.score = score
        self.features = []
        self.get_all_features()

    def __str__(self):
        feature_str = ','.join(str(x) for x in self.features)
        return ','.join([self.essay_id, self.essay_set, str(self.score), feature_str]) + '\n'

    def numerical_features(self):
        self.features.append(len(self.essay_str.split('.'))) # Number of sentences
        for sentence in self.essay_str.split('.'):
            tmp_tokens = word_tokenize(sentence)
            nltk.pos_tag(tmp_tokens)
        no_punctuation = self.essay_str.lower().translate(None, string.punctuation)
        self.tokens = nltk.word_tokenize(no_punctuation) # Yes, A new self variable. Sorry.
        self.features.append(len(self.tokens)) # Number of tokens
        self.features.append((float(len(''.join(self.tokens)))/float(len(self.tokens)))*10) # Average size of token * 10
        self.features.append(len([1 for token in self.tokens if enchant_dict.check(token) == False])) # Number of misspelled words

    def get_all_features(self):
        self.numerical_features()

def get_stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def stem_tokenize(essay):
    return get_stem_tokens(nltk.word_tokenize(essay), stemmer)

def make_points():
    for index in xrange(3,4):
        with open('training_' + str(index) + '.csv','rb') as f:
            csv_rows = list(csv.reader(f, delimiter = ','))
            out_file = open('features_' + str(index) + '.csv','w')
            for row in csv_rows:
                if row[1] == str(3):
                    p = Point(row[0], row[1], row[2], int(row[6]))
                    out_file.write(str(p))

def partition_essays():
    with open('training_set.csv', 'rb') as f:
        csv_rows = list(csv.reader(f, delimiter = ','))
        index = 1
        out_file = open('training_' + str(index) + '.csv','w')
        writer = csv.writer(out_file)
        for row in csv_rows:
            if row[1] == str(index):
                writer.writerows([row])
            else:
                out_file.close()
                index += 1
                out_file = open('training_' + str(index) + '.csv','w')
                writer = csv.writer(out_file)

if __name__=='__main__':
    partition_essays()
    make_points()
