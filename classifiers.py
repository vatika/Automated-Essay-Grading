# Copyright 2015 Abhijeet Kumar ,Anurag Ghosh, Vatika Harlalka
# Classification Techniques
# Implemented ::  Linear Regression
# Under Implementation :: SVR  , Cohen's Kappa
# To Implement ::  Graph Diffusion,etc

import numpy as np
import csv
import sklearn
import nltk
import sys

class SVR:
    pass

class Linear_Regression:
    ''' all symbols used here are a generic reresentation used in linear regression algorithms'''
    def __init__(self,X,Y):
        self.max_limit = 100000   # limit on total number  of iterations
        self.max_limit = 1000  # limit on total number  of iterations
        self.eta = 0.00001;       # approximate value of eta works good
        self.X = X
        self.Y = Y
        #dimensions (m*d) of the training set
        self.d = np.size(self.X,axis=1)
        self.m = np.size(self.X,axis=0)
        self.theta = np.zeros((self.d,1))

    def __str__(self):
        temp = ["%.10f" % x for x in self.theta]
        s =  ' '.join(temp)
        return s
        
    def calculate_cost(self):
        temp = np.matrix(self.Y-self.X.dot(self.theta))
        self.J = temp.T.dot(temp)
        
    def gradient_descent(self): 
        for i in range(self.max_limit):
            self.calculate_cost()
            P = self.X.dot(self.theta)
            update = (self.eta/self.m)*(((P-self.Y).T*self.X).T)
            #print update,i
            if abs(max(update)) < 5*(self.eta/self.m):
                break
            np.seterr(all="raise")
            self.theta = self.theta - update;

            
    def predict(self,x):
        return sum( x*self.theta)
    
    def execute(self,X_test,Y_test):
        self.gradient_descent();
        P = np.zeros(len(Y_test))
        for i in range(len(X_test)):
            P[i] = self.predict(X_test[i])
        P = np.round(P);
        return quadratic_weighted_kappa(Y_test,P, 0, 3)    

def weighted_distance(x,y):
    return abs(x-y)

def Cohen_Kappa(A,B):
    data = []
    for i in range(len(A)):
        data.append(( 'original' , i , A.item(i) ))
        data.append(( 'predicted', i , B[i] ))
#    print data    
    s = nltk.metrics.AnnotationTask(data=data,distance=weighted_distance)
    return s.weighted_kappa()#(max_distance=1.0)
#    result = np.zeros((c,c))
#    W = np.array([0,1,2,3,1,0,1,2,2,1,0,1,3,2,1,0])
#    for i in range(len(A)):
#        j = A[i]
#        k = B[i]
#        result[j,k] = result[j,k]+1;
#    a = result*W
#    b = W
#    return 1 - a/float(b);



def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
#    print rater_a.flatten()
    k = [item for sublist in rater_a for item in sublist]
    print k
    print rater_b
    sys.exit()
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)] for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        print a,b,min_rating
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat



def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    k= rater_a.flatten()
    for i in k:
        for j in i:
            print j
#    print rater_b
    sys.exit()
    for k in rater_a:
        print k[0]
    sys.exit()
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator



def data_manipulation():
    for i in range(3,4): #to change after feature extraction done for all sets
        
        # training data
        train_data = []
        with open('./Data/features_'+str(i)+'.csv','r') as in_file:
             csv_content = list(csv.reader(in_file,delimiter=','))
             for row in csv_content:
                train_data.append(row)
        
        train_data = train_data[1:]   #clip the header
        train_data = np.matrix(train_data,dtype='float64')  
        Y_train = train_data[:,2].copy()     #actual_values
        X_train = train_data[:,2:].copy()    #actual_data with random bias units
        m = np.size(X_train,axis=0)
        X_train[:,0] = np.ones((m,1)) #bias units modified
        
        #testing data
        test_data = [] # for now both are same modify here to test data
        with open('./Data/features_'+str(i)+'.csv','r') as in_file:
             csv_content = list(csv.reader(in_file,delimiter=','))
             for row in csv_content:
                test_data.append(row)
                
        test_data = test_data[1:]   #clip the header
        test_data = np.matrix(test_data,dtype='float64')
        Y_test = test_data[:,2].copy()     #actual_values       
        X_test = test_data[:,2:].copy()    #actual_data with random bias units
        m = np.size(X_test,axis=0)
        X_test[:,0] = np.ones((m,1)) #bias units modified
        
        #stroing the results for further use maybe(such as single value predictions) .......
        out_file = open('./classifier_weights/essay_set'+str(i)+'.csv','w')
        
        #Linear Regression
        L = Linear_Regression(X_train,Y_train)
        cohen_kappa_result = L.execute(X_test,Y_test)
        print cohen_kappa_result
        #other techniques coming soon
        
        writer = csv.writer(out_file,delimiter=',')
        writer.writerows([str(L).split()])
        out_file.close();
    
if __name__=='__main__':
    data_manipulation();
