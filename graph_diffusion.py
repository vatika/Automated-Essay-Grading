# graph diffusion
import csv
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from sklearn.cross_validation import KFold

class graph_diffusion():
    # similarity measure
    def similarity_measure(self,X):
        # this is an NxD matrix, where N is number of items and D its dimensionalites
        s = 100
        pairwise_dists = -1 * squareform(pdist(X, 'euclidean'))**2
        return  scipy.exp(pairwise_dists / s ** 2)

    def calculate_degree_matrix(self,W):
        return  np.diag(sum(W.T))

    # graph adjacenvy matrix formulation
    def formulate_graph_laplacian(self,X): # this is an NxD matrix
        W = self.similarity_measure(X)
        D = self.calculate_degree_matrix(W)
        # to normalize laplacian
        L = D - W
        return L

    def train(self,x_train,x_test,y_train):
        # Y   n*l(no of categories) matrix
        # a column has values 1 -1 0 for present , not present and not known
        # ghosh compute this Y
        # Y  =

        X = np.concatenate((x_train,x_test),axis=0)
        L = self.formulate_graph_laplacian(X)
        self.SVD = np.linalg.svd(L,full_matrices=1,compute_uv=1)
        [self.E_vec,self.E_val,waste] = np.diag(self.SVD[1])
        #heat matrix at different times(scales) and visualizations
        # small t for small diffusion and vice versa
        for  i in xrange(1,3)
            t =  10**i
            temp = scipy.exp(-self.E_val*t)
            H1 = np.dot(np.dot(self.E_vec,np.diag(self.E_val)),self.E_Vec.T)
            Y1 = H*Y
            # now maximum voting comes in play
        # now voting in high time and low time     

    def predict(self):
        raise BaseException("Fuck You")

class k_fold_cross_validation:
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
            stat_obj.train(x_train,x_test,y_train)
            y_pred = stat_obj.predict()
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
        diffusion_k_cross = k_fold_cross_validation(cross_valid_k,graph_diffusion,X_train,Y_train,range_min,range_max)
        print diffusion_k_cross.execute()
