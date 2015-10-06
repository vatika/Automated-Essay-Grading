# Copyright 2015 - Anurag Ghosh, Vatika Harlalka, Abhijeet Kumar
import csv
import nltk
import string
import json

from collections import Counter

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

import enchant

import math

'''
    Globals for the feature extraction. Mostly holds constants.
'''
enchant_dict = enchant.Dict("en_US")
stemmer = PorterStemmer()
# http://www.math.cornell.edu/~mec/2003-2004/cryptography/subs/frequencies.html
beauty_reference = {
    'a' : 8.12, 'b' : 1.49, 'c' : 2.71, 'd' : 4.32, 'e' : 12.02, 'f' : 2.30,
    'g' : 2.03, 'h' : 5.92, 'i' : 7.31, 'j' : 0.10, 'k' : 0.69, 'l' : 3.98,
    'm' : 2.61, 'n' : 6.95, 'o' : 7.68, 'p' : 1.82, 'q' : 0.11, 'r' : 6.02,
    's' : 1.68, 't' : 9.10, 'u' : 2.88, 'v' : 1.11, 'w' : 2.09, 'x' : 0.17,
    'y' : 2.11, 'z' : 0.07,
}
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
pos_classes = [ [ "CC","DT","EX","IN","MD","TO","UH","PDT","POS" ], [ "FW","CD","LS","RP","SYM" ], \
                [ "JJ","JJR","JS" ], [ "NN","NNS","NNP", "NNPS" ], [ "PRP","PRP$" ], \
                [ "RB","RBR","RBS" ], [ "VBD","VBG","VBN","VBP","VBZ" ], [ "WDT","WP","WP$","WRB" ],
              ]
# http://crr.ugent.be/archives/806
words_acq_age = json.load(open('aoa_values.json'))

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
        self.stat_sentences = []
        self.std_nword = 0
        self.mean_nword = 0

    def __str__(self):
        '''
        return all the features along with the idendity of the essay (essay set , essay id etc)
        '''
        feature_str = ','.join(str(x) for x in self.features)
        return ','.join([self.essay_id, self.essay_set, str(self.score), feature_str, str(self.bag_of_words)]) + '\n'

    def get_label(self):
        string = "id,set,human_score,sentence_count,word_count,avg_word_length,misspell_words,char_4,char_6,char_8,char_10,char_12,"
        string += "mean_char,std_char,word_10,word_18,word_25,mean_word,std_word,pos_conj,pos_misc,pos_adj,pos_noun,pos_prep,pos_adv,pos_vrb,pos_wh,"
        string += "beauty_score,vocabulory_score,maturity_score,bag_of_words_score\n"
        return string

    def numerical_features(self):
        '''
        numerical features as number of tokens , sentences , misspells, number of words of different character lengths, average word length, standard devaiation of word length
        '''
        sentences = self.essay_str.split('.')
        self.features.append(len(sentences)) # Number of sentences
        no_punctuation = self.essay_str.lower().translate(None, string.punctuation)
        self.tokens = nltk.word_tokenize(no_punctuation) # Yes, A new self variable. Sorry.
        self.features.append(len(self.tokens)) # Number of tokens
        self.features.append((float(len(''.join(self.tokens)))/float(len(self.tokens)))*10) # Average size of token * 10
        self.features.append(len([1 for token in self.tokens if enchant_dict.check(token) == False])) # Number of misspelled words

        len_words = []
        stat_words = [0,0,0,0,0] #Number of words with character length > 4,6,8,10,12 respectively
        for token in self.tokens:
            l = len(token)
            len_words.append(l)
            if l > 4:
                if l > 6:
                    if l > 8:
                        if l > 10:
                            if l > 12:
                                stat_words[4] += 1
                            stat_words[3] += 1
                        stat_words[2] += 1
                    stat_words[1] += 1
                stat_words[0] += 1
        for stat in stat_words:
            self.features.append(stat)
        mean =  sum(len_words)/float(len(self.tokens))
        self.features.append(mean)
        self.features.append(math.sqrt(sum([pow(x-mean,2) for x in len_words])/float(len(len_words)-1)))
        stat_sentences = [0, 0, 0] #Number of sentences with word length > 10, 18, 25
        len_sentences = []
        for sentence in sentences:
            n_words = len(nltk.word_tokenize(sentence.lower().translate(None, string.punctuation)))
            len_sentences.append(n_words)
            if n_words > 10:
                if n_words > 18:
                    if n_words > 25:
                        stat_sentences[2] += 1
                    stat_sentences[1] += 1
                stat_sentences[0] += 1
        for stat in stat_sentences:
            self.features.append(stat)
        mean =  sum(len_sentences)/float(len(sentences))
        self.features.append(mean)
        self.features.append(math.sqrt(sum([pow(x-mean,2) for x in len_sentences])/float(len(len_sentences))))

            

    def stylized_word_scores(self):
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
        self.maturity_score = 0
        vocab = 0
        for word in words:
            lower_word = word.lower()
            if lower_word in words_acq_age and len(lower_word) > 3:
                self.maturity_score = self.maturity_score + float(words_acq_age[lower_word])
                vocab += 1
        self.maturity_score /= vocab
        self.features.append(vocab)
        self.features.append(self.maturity_score)

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
        self.stylized_word_scores()

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
            values = Bag_of_Words(essay_tokens, all_words)
            for i in xrange(len(points)):
                points[i].set_bag_of_words(values[i])
            out_file = open('features_' + str(index) + '.csv','w')
            out_file.write(points[0].get_label())
            for p in points:
                out_file.write(str(p))

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
