# Copyright 2015 Abhijeet Kumar ,Anurag Ghosh, Vatika Harlalka
# Classification Techniques
# Implemented ::  Linear Regression
# To Implement :: SVM , Graph Diffusion,etc

import numpy as np
import csv

class Linear_Regression:
    ''' all symbols used here are a generic reresentation used in linear regression algorithms'''
    def __init__(self,X,Y):
        self.max_limit = 10000   # limit on total number  of iterations
        self.eta = 0.0001;       # approximate value of eta works good
        self.X = X
        self.Y = Y
        #dimensions (m*d) of the training set
        self.d = np.size(self.X,axis=1)
        self.m = np.size(self.X,axis=0)
        self.theta = np.zeros((self.d,1));

    def __str__(self):
        temp = ["%.10f" % x for x in self.theta]
        s =  ' '.join(temp)
        return s
        
        
    def calculate_cost(self):
        temp = np.matrix(self.Y-self.X.dot(self.theta))
        self.J = temp.T.dot(temp)
        
    def gradient_descent(self): 
        for i in range(self.max_limit):
            #cost function
            self.calculate_cost()
            #print J
            #print theta
            if self.J < 1:
                break
            P = self.X.dot(self.theta)
            # print np.size(P)
            #theta update_step
            self.theta = self.theta - (self.eta/self.m)*(((P-self.Y).T*self.X).T);
            
        #print theta
    def execute(self):
        self.gradient_descent();


def data_manipulation():
    for i in range(3,4): #to change after feature extraction done for all sets
        data = []
        with open('./Data/features_'+str(i)+'.csv','r') as in_file:
             csv_content = list(csv.reader(in_file,delimiter=','))
             for row in csv_content:
                data.append(row)
        
        data = data[1:]   #clip the header
        data = np.matrix(data,dtype='float64')  
        Y = data[:,2]     #actual_values
        X = data[:,2:]    #actual_data with random bias units
        m = np.size(X,axis=0)
        X[:,0] = np.ones((m,1)) #bias units modified
        
        out_file = open('./classifier_weights/essay_set'+str(i)+'.csv','w')
        
        L = Linear_Regression(X,Y)
        L.execute()
        #other techniques coming soon
        
        writer = csv.writer(out_file,delimiter=',')
        writer.writerows([str(L).split()])
        out_file.close();
    
if __name__=='__main__':
    data_manipulation();
