# Copyright 2015 - Anurag Ghosh, Vatika Harlalka, Abhijeet Kumar
import csv
import nltk
import string

from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

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
        no_punctuation = self.essay_str.lower().translate(None, string.punctuation)
        tokens = nltk.word_tokenize(no_punctuation)
        self.features.append(len(tokens)) # Number of tokens

    def get_all_features(self):
        self.numerical_features()

def make_points():
    for index in xrange(3,4):
        with open('training_' + str(index) + '.csv','rb') as f:
            csv_rows = list(csv.reader(f, delimiter = ','))
            out_file = open('features_' + str(index) + '.csv','w')
            for row in csv_rows:
                if row[1] == str(3):
                    p = Point(row[0],row[1],row[2], int(row[6]))
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
