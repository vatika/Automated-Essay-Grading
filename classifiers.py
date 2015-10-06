# Copyright 2015 Abhijeet Kumar ,Anurag Ghosh, Vatika Harlalka
# Classification Techniques
# Implemented ::  Linear Regression
# Under Implementation :: SVR  , Cohen's Kappa
# To Implement ::  Graph Diffusion,etc

import numpy as np
import csv
import sklearn
import nltk
import weighted_kappa as own_wp

class SVR:
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

class Linear_Regression:
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
        L = Linear_Regression()
        L.train(X_train,Y_train)
        cohen_kappa_result = L.find_kappa(X_test,Y_test)
        print 'Linear Regression = ' +str(cohen_kappa_result)
        
        #SVR
        M = SVR()
        M.train(X_train,Y_train)
        cohen_kappa_result = M.find_kappa(X_test,Y_test)
        print 'SVR = '+str(cohen_kappa_result)
        
        #other techniques coming soon

        writer = csv.writer(out_file,delimiter=',')
        writer.writerows([str(L).split()])
        out_file.close();

if __name__=='__main__':
    data_manipulation();
