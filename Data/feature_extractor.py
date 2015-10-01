# Copyright 2015 - Anurag Ghosh, Vatika Harlalka, Abhijeet Kumar
import csv
import nltk
import string

from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

class Point:
    def __init__(self,essay_id,essay_set,essay_str,score):
        this.essay_id = essay_id
        this.essay_set = essay_set
        this.essay_str = essay_str
        this.score = score
        this.features = []

    def __str__():
        feature_str = ','.join(this.features)
        return ','.join([this.essay_id, this.essay_set, str(this.score), feature_str])

    def numerical_features():
        this.features.append(len(this.essay_str.split('.'))) # Number of sentences
        no_punctuation = this.essay_str.lower().translate(None, string.punctuation)
        tokens = nltk.word_tokenize(no_punctuation)
        this.features.append(len(tokens)) # Number of tokens


def partition_essays():
    with open('training_set.csv', 'rb') as f:
        csv_rows = list(csv.reader(f, delimiter = ','))
        i = 1
        out_file = open('training_' + str(i) + '.csv','w')
        writer = csv.writer(out_file)
        for row in csv_rows:
            if row[1] == str(i):
                writer.writerows([row])
            else:
                out_file.close()
                i += 1
                out_file = open('training_' + str(i) + '.csv','w')
                writer = csv.writer(out_file)

if __name__=='__main__':
    partition_essays()
