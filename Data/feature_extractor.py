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
        self.all_features()
        self.bag_of_words = 0
    # return all the features along with the idendity of the essay (essay set , essay id etc)
    def __str__(self):
        feature_str = ','.join(str(x) for x in self.features)
        return ','.join([self.essay_id, self.essay_set, str(self.score), feature_str]) + '\n'

    #numerical features as number of tokens , sentences , misspells ...
    def numerical_features(self):
        self.features.append(len(self.essay_str.split('.'))) # Number of sentences
        no_punctuation = self.essay_str.lower().translate(None, string.punctuation)
        self.tokens = nltk.word_tokenize(no_punctuation) # Yes, A new self variable. Sorry.
        self.features.append(len(self.tokens)) # Number of tokens
        self.features.append((float(len(''.join(self.tokens)))/float(len(self.tokens)))*10) # Average size of token * 10
        self.features.append(len([1 for token in self.tokens if enchant_dict.check(token) == False])) # Number of misspelled words
    
    #parts of speech(noun adjectives verbs ....) counts
    def pos_features(self):
        s = {}
        for sentence in self.essay_str.split('.'):
            try:
                tmp_tokens = nltk.word_tokenize(sentence)
                values = nltk.pos_tag(tmp_tokens)
                for v in values:
                    if v[1] in s:
                        s[v[1]] += 1
                    else:
                        s[v[1]] = 1
            except UnicodeDecodeError:
                continue
            except IndexError:
                continue
        # This is occurance of each kind of part of speech in the essay.
        # We need to find a simple/complex way to classify them and convert them into feature values.
        #print s
    
    # compute all the features for the essay
    # to add spell unigrams and n-grams 
    def all_features(self):
        self.numerical_features()
        self.pos_features()
    def set_bag_of_words(self, v):
        self.bag_of_words = v
#not in use right now
def get_stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

#not in use right now
def stem_tokenize(essay):
    return get_stem_tokens(nltk.word_tokenize(essay), stemmer)

def Bag_of_Words(essay_tokens, all_words):
    
    count = Counter(all_words)
    common_words = [x[0] for x in count.most_common(10)]

    essays  = [" ".join([str(word) for word in e if word in common_words]) for e in essay_tokens]
    
    tfidf = TfidfVectorizer(tokenizer=stem_tokenize, stop_words='english')
    tfs = tfidf.fit_transform(essays)

    feature_names = tfidf.get_feature_names()

    v = []
    for i in xrange(len(essay_tokens)):
       value = 0
       count_essay = Counter(essay_tokens[i])
       common_w_essay = count_essay.most_common(10)
       common_words_essay = [word[0] for word in common_w_essay]
       for j in xrange(len(common_words_essay)):
           if common_words_essay[j] in common_words:
               for col in tfs.nonzero()[1]:
                  if feature_names[col] == common_words_essay[j]:
                           value += common_w_essay[j][1]*tfs[0,col]
                           break
       v.append(value)
    return v
#treating each essay as a (data)point in the essay set and extracting all the features
def make_points():
    # currently performing tasks on essay set 3 only
    for index in xrange(3,4):
        with open('training_' + str(index) + '.csv','rb') as f:
            all_words = []
            essays = []
            csv_rows = list(csv.reader(f, delimiter = ','))
            out_file = open('features_' + str(index) + '.csv','w')
            essay_tokens = []
            points = []
            for row in csv_rows:
                # currently performing tasks on essay set 3 only
                if row[1] == str(3):
                   tokens = stem_tokenize(row[2].translate(None, string.punctuation).decode('utf-8'))
                   essay_tokens.append(tokens)
                   all_words.extend(tokens)
                   p = Point(row[0], row[1], row[2], int(row[6]))
                   points.append(p)
                   out_file.write(str(p))
            values = Bag_of_Words(essay_tokens, all_words)
            for i in xrange(len(points)):
                points[i].set_bag_of_words(values[i])
             
                    
#partition the training essay sets into different csv files
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
