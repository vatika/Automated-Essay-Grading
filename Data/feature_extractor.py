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

import urllib2
from bs4 import BeautifulSoup

enchant_dict = enchant.Dict("en_US")
beauty_reference = {}
stemmer = PorterStemmer()
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
pos_classes = [ [ "CC","DT","EX","IN","MD","TO","UH","PDT","POS" ], [ "FW","CD","LS","RP","SYM" ], \
                [ "JJ","JJR","JS" ], [ "NN","NNS","NNP", "NNPS" ], [ "PRP","PRP$" ], \
                [ "RB","RBR","RBS" ], [ "VBD","VBG","VBN","VBP","VBZ" ], [ "WDT","WP","WP$","WRB" ],
              ]

class Point:
    def __init__(self,essay_id,essay_set,essay_str,score):
        '''
        We store the id, set_number, essay, actual score and the features
        '''
        self.essay_id = essay_id
        self.essay_set = essay_set
        self.essay_str = essay_str
        self.score = score
        self.features = []
        self.all_features()
        self.bag_of_words = 0

    def __str__(self):
        '''
        return all the features along with the idendity of the essay (essay set , essay id etc)
        '''
        feature_str = ','.join(str(x) for x in self.features)
        return ','.join([self.essay_id, self.essay_set, str(self.score), feature_str, str(self.bag_of_words)]) + '\n'

    def numerical_features(self):
        '''
        numerical features as number of tokens , sentences , misspells
        '''
        self.features.append(len(self.essay_str.split('.'))) # Number of sentences
        no_punctuation = self.essay_str.lower().translate(None, string.punctuation)
        self.tokens = nltk.word_tokenize(no_punctuation) # Yes, A new self variable. Sorry.
        self.features.append(len(self.tokens)) # Number of tokens
        self.features.append((float(len(''.join(self.tokens)))/float(len(self.tokens)))*10) # Average size of token * 10
        self.features.append(len([1 for token in self.tokens if enchant_dict.check(token) == False])) # Number of misspelled words


    def beautiful_word_score(self):
        words = stem_tokenize(self.essay_str.translate(None, string.punctuation).decode('utf-8'))
        self.beauty_score = 0
        for word in words:
            s = 1.0
            for letter in word:
                try:
                    s = s*beauty_reference[letter.lower()]
                except:
                    pass
            self.beauty_score += 1/s
        self.features.append(self.beauty_score)

    def pos_features(self):
        '''
        parts of speech(noun adjectives verbs ....) counts
        '''
        s = {}
        freqs = [0 for i in xrange(0,len(pos_classes))]
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
        for key,value in s.iteritems():
             for index, pos in enumerate(pos_classes):
                 if key in pos:
                     freqs[index] += value
                     break
        self.features.extend(freqs)

    def set_bag_of_words(self, v):
        '''
        bag of words value, the score is a linear combination
        of the frequencies of most common words and associated weights
        '''
        self.bag_of_words = v

    def all_features(self):
        '''
        compute all the features for the essay
        to add spell unigrams and n-grams
        '''
        self.numerical_features()
        self.pos_features()
        self.beautiful_word_score()

def get_stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

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

def make_points():
    '''
    treating each essay as a (data)point in the essay set and extracting all the features
    '''
    # currently performing tasks on essay set 3 only
    for index in xrange(3,4):
        with open('training_' + str(index) + '.csv','rb') as f:
            all_words = []
            essays = []
            csv_rows = list(csv.reader(f, delimiter = ','))
            out_file = open('features_' + str(index) + '.csv','w')
            essay_tokens = []
            points = []
            get_beauty_table()
            for row in csv_rows:
                # currently performing tasks on essay set 3 only
                if row[1] == str(3):
                   tokens = stem_tokenize(row[2].translate(None, string.punctuation).decode('utf-8'))
                   essay_tokens.append(tokens)
                   all_words.extend(tokens)
                   p = Point(row[0], row[1], row[2], int(row[6]))
                   points.append(p)
            values = Bag_of_Words(essay_tokens, all_words)
            for i in xrange(len(points)):
                points[i].set_bag_of_words(values[i])
            for p in points:
                out_file.write(str(p))

def get_beauty_table():
    #Get reference table
    contenturl = "http://www.math.cornell.edu/~mec/2003-2004/cryptography/subs/frequencies.html"
    soup = BeautifulSoup(urllib2.urlopen(contenturl).read())
    global beauty_reference
    table = soup.find("table")
    rows = table.findAll('tr')
    beauty_reference = {}
    f = 0
    for row in rows:
            cols = row.findAll('td')
            a,b,c,d,e = [c.text for c in cols]
            if f:
                    beauty_reference[d.encode('utf-8').lower()] = float(e.encode('utf-8'))
            f = True



def partition_essays():
    '''
    partition the training essay sets into different csv files
    '''
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
