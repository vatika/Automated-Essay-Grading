#Copyright 2015 Abhijeet Kumar ,Anurag Ghosh, Varika Harlalka

import numpy as np
import csv
np.set_printoptions(suppress=True)

max_limit = 10000 # limit on total number  of iterations
eta = 0.0001; # approximate value of eta works good

''' all symbols used here are a generic reresentation used in linear regression algorithms'''
data = []
with open('./features_3.csv','r') as in_file:
     csv_content = list(csv.reader(in_file,delimiter=','))
     for row in csv_content:
        data.append(row)

data = data[1:]
data = np.matrix(data,dtype='float64')
Y = data[:,2]
X = data[:,2:]

d = np.size(X,axis=1)
m = np.size(X,axis=0)

X[:,0] = np.ones((m,1))
theta = np.zeros((d,1))

for i in range(max_limit):
    temp = np.matrix(Y-X.dot(theta))
    J = temp.T.dot(temp)

    #print J
    #print theta
    if J < 1:
        break
    P = X.dot(theta)
   # print np.size(P)
    theta = theta - (eta/m)*(((P-Y).T*X).T);
    
print theta