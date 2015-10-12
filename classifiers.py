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
    from sklearn.svm import SVR, SVC
    from sklearn.linear_model import LinearRegression
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier
    import csv
    import nltk
    import weighted_kappa as own_wp
    import random

class meta_classifier(object):
    ''' all symbols used here are a generic reresentation used in support vector regression algorithms'''
    def __init__(self, learner):
        self.L = learner

    def __str__(self):
        return self.L.__str__();

    def train(self,X_train,Y_train):
        temp = []
        for i in range(len(Y_train)):
            temp.append(Y_train.item(i))
        self.L.fit(X_train,temp)

    def predict(self,X_test):
        d = self.L.predict(X_test)
        return d

class support_vector_regression(meta_classifier):
    def __init__(self):
        super(self.__class__, self).__init__(SVR(kernel='rbf', gamma=0.00003))

class kernel_ridge_regression(meta_classifier):
    def __init__(self):
        super(self.__class__, self).__init__(KernelRidge(kernel='rbf', gamma=0.00003, alpha=0.625)) # alpha = 1/(2*C)

class support_vector_machine(meta_classifier):
    def __init__(self):
        super(self.__class__, self).__init__(SVC(kernel='rbf', gamma=0.0004, C=0.8))

class decision_tree_classifier(meta_classifier):
    def __init__(self):
        super(self.__class__, self).__init__(DecisionTreeClassifier(criterion='entropy'))

class linear_regression:
    ''' all symbols used here are a generic reresentation used in linear regression algorithms'''
    def __init__(self):
        self.L = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)

    def __str__(self):
        return self.L.__str__();

    def train(self,X_train,Y_train):
        self.L.fit(X_train,Y_train)

    def predict(self,X_test):
        return self.L.predict(X_test)[0][0]

class k_fold_cross_validation:
    '''
        The class will take an statistical class and training set and parameter k.
        The set will be divided wrt to k and cross validated using the statistical
        class provided.
        The statistical class should have two methods and no constructor args -
        method train(training_x, training_y)
        method predict(x_test_val)
    '''
    def __init__(self,k,stat_class,x_train,y_train,range_min,range_max):
        self.k_cross = float(k)
        self.stat_class = stat_class
        self.x_train = x_train
        self.y_train = y_train
        self.values = []
        self.range_min = range_min
        self.range_max = range_max

    def execute(self):
        kf = KFold(len(self.x_train), n_folds=self.k_cross)
        own_kappa = []
        for train_idx, test_idx in kf:
            x_train, x_test = self.x_train[train_idx], self.x_train[test_idx]
            y_train, y_test = self.y_train[train_idx], self.y_train[test_idx]
            stat_obj = self.stat_class() # reflection bitches
            stat_obj.train(x_train,y_train)
            y_pred = [ 0 for i in xrange(len(y_test)) ]
            for i in range(len(x_test)):
                val = int(np.round(stat_obj.predict(x_test[i])))
                if val > self.range_max: val = self.range_max
                if val < self.range_min: val = self.range_min
                y_pred[i] = [val]
            y_pred = np.matrix(y_pred)
            cohen_kappa_rating = own_wp.quadratic_weighted_kappa(y_test,y_pred,self.range_min,self.range_max)
            self.values.append(cohen_kappa_rating)
        print str(sum(self.values)/self.k_cross)


def data_manipulation():
    for i in [1,3,4,5,6]: #to change after feature extraction done for all sets
        # training data
        train_data = []
        with open('./Data/features_'+str(i)+'.csv','r') as in_file:
             csv_content = list(csv.reader(in_file,delimiter=','))
             for row in csv_content:
                train_data.append(row)
        header = train_data[0]
        train_data = train_data[1:]   #clip the header
        train_data = np.matrix(train_data,dtype='float64')
        Y_train = train_data[:,2].copy()     #actual_values
        X_train = train_data[:,2:].copy()    #actual_data with random bias units
        m = np.size(X_train,axis=0)
        X_train[:,0] = np.ones((m,1)) #bias units modified
        cross_valid_k = 5
        range_max = range_min = 0
        if i == 1:
            range_min = 2
            range_max = 12
        elif i == 3 or i == 4:
            range_max = 3
        elif i == 5 or i == 6:
            range_max = 4
        linear_k_cross = k_fold_cross_validation(cross_valid_k,linear_regression,X_train,Y_train,range_min,range_max)
        print str(i) + " linear_regression :\t\t\t\t\t",
        linear_k_cross.execute()
        svr_k_cross = k_fold_cross_validation(cross_valid_k,support_vector_regression,X_train,Y_train,range_min, range_max)
        print str(i) + " support_vector_regression :\t\t\t\t",
        svr_k_cross.execute()
        svm_k_cross = k_fold_cross_validation(cross_valid_k,support_vector_machine,X_train,Y_train, range_min, range_max)
        print str(i) + " support_vector_machine :\t\t\t\t",
        svm_k_cross.execute()
        kernel_regress_k_cross = k_fold_cross_validation(cross_valid_k,kernel_ridge_regression,X_train,Y_train, range_min, range_max)
        print str(i) + " kernel_ridge_regression :\t\t\t\t",
        kernel_regress_k_cross.execute()
        decision_class_k_cross = k_fold_cross_validation(cross_valid_k,decision_tree_classifier,X_train,Y_train, range_min, range_max)
        print str(i) + " decision_tree_classifier :\t\t\t\t",
        decision_class_k_cross.execute()

if __name__=='__main__':
    data_manipulation();
