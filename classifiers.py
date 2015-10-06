# Copyright 2015 Abhijeet Kumar ,Anurag Ghosh, Vatika Harlalka
# Classification Techniques
# Implemented ::  Linear Regression
# Under Implementation :: SVR  , Cohen's Kappa
# To Implement ::  Graph Diffusion,etc

import numpy as np
import csv
import sklearn
import nltk

class SVR:
    pass

class Linear_Regression:
    ''' all symbols used here are a generic reresentation used in linear regression algorithms'''
    def __init__(self,X,Y):
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
            print update,i
            if abs(max(update)) < 5*(self.eta/self.m):
                break
            self.theta = self.theta - update;
            
    def predict(self,x):
        return sum( x*self.theta)
    
    def execute(self,X_test,Y_test):
        self.gradient_descent();
        P = np.zeros(len(Y_test))
        for i in range(len(X_test)):
            P[i] = self.predict(X_test[i])
        P = np.round(P);
        return Cohen_Kappa(Y_test,P)    

def weighted_distance(x,y):
    return 1/float(4)*abs(x-y)

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