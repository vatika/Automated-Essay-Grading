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
    import csv
    import nltk
    import weighted_kappa as own_wp


class support_vector_regression:
    ''' all symbols used here are a generic reresentation used in linear regression algorithms'''
    def __init__(self):
        self.L = sklearn.svm.SVR(kernel='rbf', degree=3, gamma=0.1)

    def __str__(self):
        return self.L.__str__();

    def train(self,X_train,Y_train):
        temp = []
        for i in range(len(Y_train)):
            temp.append(Y_train.item(i))
        self.L.fit(X_train,temp)

    #prediction for a single value only to be used later(maybe)
    def predict(self,X_test):
        return self.L.predict(X_test)

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

    #prediction for a single value only to be used later(maybe)
    def predict(self,X_test):
        return self.L.predict(X_test)
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
    '''
    def __init__(self,k,stat_class,x_train,y_train):
        self.k = k
        self.stat_class = stat_class
        self.x_train = x_train
        self.y_train = y_train

    def execute(self):
        stat_obj = self.stat_class() # reflection bitches
        stat_obj.train(self.x_train,self.y_train)
        cohen_kappa_result = stat_obj.find_kappa(self.x_train,self.y_train)
        print str(stat_obj) + str(cohen_kappa_result)


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
        linear_k_cross.execute()
        svr_k_cross = k_fold_cross_validation(cross_valid_k,support_vector_regression,X_train,Y_train)
        svr_k_cross.execute()

if __name__=='__main__':
    data_manipulation();
