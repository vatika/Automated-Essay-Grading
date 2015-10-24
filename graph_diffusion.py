# Copyright 2015 Abhijeet Kumar ,Anurag Ghosh, Vatika Harlalka
# Graph Diffusion Techniques

import warnings

with warnings.catch_warnings():
    import csv

    import numpy as np

    import scipy
    import scipy.linalg

    from scipy.sparse import csgraph
    from scipy.spatial.distance import pdist, squareform

    from sklearn.cross_validation import KFold
    from sklearn.metrics.pairwise import chi2_kernel, rbf_kernel

    import weighted_kappa as own_wp

    from random import random
    from bisect import bisect


# similarity measures
def gaussian_kernel(X):
    # this is an NxD matrix, where N is number of items and D its dimensionalites
    s = 100
    pairwise_dists = -1 * squareform(pdist(X, 'euclidean'))**2
    return  scipy.exp(pairwise_dists / s ** 2)

def linear_kernel(X):
    pairwise_dists = 1/squareform(pdist(X, 'euclidean'))
    nan_idx = np.isnan(pairwise_dists)
    pairwise_dists[pairwise_dists == -np.inf] = 0.0
    pairwise_dists[pairwise_dists == +np.inf] = 0.0
    pairwise_dists[nan_idx] = 0.0
    #print np.shape(pairwise_dists)
    return pairwise_dists

class graph_diffusion():
    def __init__(self,range_min,range_max, similarity_measure,voting="exponential"):
        self.range_min = range_min
        self.range_max = range_max
        self.similarity_measure = similarity_measure
        self.voting = voting

    def calculate_degree_matrix(self,W):
        return np.diag(sum(W.T))

    # graph adjacenvy matrix formulation
    def formulate_graph_laplacian(self,X): # this is an NxD matrix
        W = self.similarity_measure(X)
        D = self.calculate_degree_matrix(W)
        return csgraph.laplacian(W, normed=True), D

    # The formulation is transducive,
    # ie. the training set and the test
    # set is known.
    def train(self,x_train,x_test,y_train):
        # Y   n*l(no of categories) matrix
        # a column has values 1 -1 0 for present , not present and not known
        self.test_size = len(x_test)
        self.train_size = len(y_train)
        self.dim = self.range_max - self.range_min + 1
        self.Y = np.zeros((len(x_train)+len(x_test), self.dim))
        rng = [val for val in xrange(0, self.dim)]
        for itx in xrange(0, len(y_train)):
            self.Y[itx, rng] = -1
            self.Y[itx,int(y_train[itx])-range_min] = 1
        self.X = np.concatenate((x_train,x_test),0)
        self.L,self.D = self.formulate_graph_laplacian(self.X)
        [self.E_val,self.E_vec_U] = scipy.linalg.eigh(self.L,self.D)

    def exp_voting(self,Z):
        ans = np.zeros(self.test_size)
        weighted_denom = 0
        for i in xrange(0,itr):
            weighted_denom += scipy.exp(-1*i)
            ans += scipy.exp(-1*i)*Z[:,i]
        return np.round(ans/weighted_denom)

    def average_voting(self,Z):
        return np.round(sum(Z.T)/itr)

    def stochastic_voting(self, Z):
        weights = [scipy.exp(-1*i) for i in xrange(0,self.itr)]
        wieghts = weights/sum(weights)
        ans = np.zeros(self.test_size)
        for i in xrange(0,self.test_size):
            ans[i] = np.random.choice(Z[i,:], p=weights)
        return ans

    def predict(self):
        # heat matrix at different times(scales) and visualizations
        # small t for small diffusion and vice versa
        self.itr = 5
        Z = np.zeros((self.test_size,self.itr))
        for i in xrange(0,self.itr):
            # the main trick is in the following 5 lines
            t =  0.000000001*(100**i)
            temp = scipy.exp(-self.E_val*t)
            H1 = np.dot(np.dot(self.E_vec_U,np.diag(temp)),self.E_vec_U.T)
            Y1 = np.dot(H1,self.Y) # matrix multiplication is so shitty in numpy/python
            # now maximum voting comes in play
            Z1 = np.zeros(self.test_size)
            for j in xrange(self.train_size,len(Y1)):
                present_max = -100000000
                max_ind = 0
                for k in xrange(0,self.dim):
                    if Y1[j,k] > present_max:
                        present_max = Y1[j,k]
                        max_ind = k
                    Z1[j - self.train_size] = max_ind + self.range_min
            Z[:,i] = Z1
        # now voting in high time and low time and prediction of scores accordingly
        if self.voting == "exponential":
            return self.exp_voting(Z)
        elif self.voting == "average":
            return self.average_voting(Z)
        elif self.voting == "stochastic":
            return self.stochastic_voting(Z)
        else:
            raise BaseException("Unsupported Voting Measure")

class k_fold_cross_validation(object):
    def __init__(self,k,stat_class,x_train,y_train,range_min,range_max,similarity_measure):
        self.k_cross = float(k)
        self.stat_class = stat_class
        self.x_train = x_train
        self.y_train = y_train
        self.values = []
        self.range_min = range_min
        self.range_max = range_max
        self.similarity_measure = similarity_measure

    def execute(self):
        kf = KFold(len(self.x_train), n_folds=self.k_cross)
        own_kappa = []
        for train_idx, test_idx in kf:
            x_train, x_test = self.x_train[train_idx], self.x_train[test_idx]
            y_train, y_test = self.y_train[train_idx], self.y_train[test_idx]
            stat_obj = self.stat_class(range_min,range_max, self.similarity_measure,voting="stochastic") # reflection bitches
            stat_obj.train(x_train,x_test,y_train)
            y_pred = np.matrix(stat_obj.predict()).T
            cohen_kappa_rating = own_wp.quadratic_weighted_kappa(y_test,y_pred,self.range_min,self.range_max)
            self.values.append(cohen_kappa_rating)
        return str(sum(self.values)/self.k_cross)



if __name__ == "__main__":
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
        diffusion_k_cross = k_fold_cross_validation(cross_valid_k,graph_diffusion,X_train,Y_train,range_min,range_max,gaussian_kernel)
        print diffusion_k_cross.execute()
