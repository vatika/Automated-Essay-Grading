# Copyright 2015 Abhijeet Kumar ,Anurag Ghosh, Vatika Harlalka
# Classification Techniques
# Implemented ::  Linear Regression
# Under Implementation :: SVR  , Cohen's Kappa
# To Implement ::  Graph Diffusion,etc

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import numpy as np
    import sklearn
    from sklearn.cross_validation import KFold
    import csv
    import nltk
    import weighted_kappa as own_wp
    import random


class support_vector_regression:
    ''' all symbols used here are a generic reresentation used in linear regression algorithms'''
    def __init__(self):
        self.L = sklearn.svm.SVR(kernel='rbf', degree=3, gamma=0.00005)

    def __str__(self):
        return self.L.__str__();

    def train(self,X_train,Y_train):
        temp = []
        for i in range(len(Y_train)):
            temp.append(Y_train.item(i))
        #print temp
        self.L.fit(X_train,temp)

    def predict(self,X_test):
        d = self.L.predict(X_test)
        #print d
        return d

    def find_kappa(self,X_test,Y_test):
        P = self.L.predict(X_test)
        P = np.zeros(len(Y_test))
        for i in range(len(X_test)):
            P[i] = self.predict(X_test[i])
        P = np.round(P)
        #take care of the fact that value greater than 3 is unaccepetable
        for i in range(len(P)):
            if P[i] > 3:
                P[i] = 3
        return own_wp.quadratic_weighted_kappa(Y_test,P, 0, 3)

class linear_regression:
    ''' all symbols used here are a generic reresentation used in linear regression algorithms'''
    def __init__(self):
        self.L = sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True)

    def __str__(self):
        return self.L.__str__();

    def train(self,X_train,Y_train):
        self.L.fit(X_train,Y_train)

    def predict(self,X_test):
        return self.L.predict(X_test)[0][0]

    def find_kappa(self,X_test,Y_test):
        P = self.L.predict(X_test)
        P = np.zeros(len(Y_test))
        for i in range(len(X_test)):
            P[i] = self.predict(X_test[i])
        P = np.round(P)
        #take care of the fact that value greater than 3 is unaccepetable
        for i in range(len(P)):
            if P[i] > 3:
                P[i] = 3
        return own_wp.quadratic_weighted_kappa(Y_test,P, 0, 3)

class k_fold_cross_validation:
    '''
        The class will take an statistical class and training set and parameter k.
        The set will be divided wrt to k and cross validated using the statistical
        class provided.
        The statistical class should have two methods and no constructor args -
        method train(training_x, training_y)
        method find_kappa(test_x,test_y)
    '''
    def __init__(self,k,stat_class,x_train,y_train):
        self.k_cross = float(k)
        self.stat_class = stat_class
        self.x_train = x_train
        self.y_train = y_train
        self.values = []

    def execute(self):
        kf = KFold(len(self.x_train), n_folds=self.k_cross)
        own_kappa = []
        for train_idx, test_idx in kf:
            x_train, x_test = self.x_train[train_idx], self.x_train[test_idx]
            y_train, y_test = self.y_train[train_idx], self.y_train[test_idx]
            stat_obj = self.stat_class() # reflection bitches
            stat_obj.train(x_train,y_train)
            y_pred = [ 0 for i in xrange(len(y_test))]
            for i in range(len(x_test)):
                val = int(np.round(stat_obj.predict(x_test[i])))
                if val > 3: val = 3
                if val < 0: val = 0
                y_pred[i] = [val]
            #print y_pred
            y_pred = np.matrix(y_pred)
            cohen_kappa_rating = own_wp.quadratic_weighted_kappa(y_test,y_pred,0,3)
            self.values.append(cohen_kappa_rating)
        print str(sum(self.values)/self.k_cross)


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

        cross_valid_k = 5
        linear_k_cross = k_fold_cross_validation(cross_valid_k,linear_regression,X_train,Y_train)
        print "linear_regression :\t\t\t\t\t",
        linear_k_cross.execute()
        svr_k_cross = k_fold_cross_validation(cross_valid_k,support_vector_regression,X_train,Y_train)
        print "support_vector_regression :\t\t\t\t",
        svr_k_cross.execute()

if __name__=='__main__':
    data_manipulation();
