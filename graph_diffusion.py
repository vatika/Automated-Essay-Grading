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

from sklearn.lda import LDA

# similarity measures
# Remember --> similarity is inversely proportional to distance measure
def gaussian_kernel(X):
    """
        Input:
            X --> The feature set of essays, of size N * D
        Output:
            exp(-|x - y|**2) for all x and y belonging to rows(X)
            This is the guassian similarity measure.
    """
    s = 100
    pairwise_dists = -1 * squareform(pdist(X, 'euclidean'))**2
    return  scipy.exp(pairwise_dists / s ** 2)

def linear_kernel(X):
    """
        Input:
            X --> The feature set of essays, of size N * D
        Output:
            1/|x - y| for all x and y belonging to rows(X)
            This is the linear similarity measure.
        Note:
            All output[i,j] equal to NaN and inf are set to 0.0
    """
    pairwise_dists = 1/squareform(pdist(X, 'euclidean'))
    nan_idx = np.isnan(pairwise_dists)
    pairwise_dists[pairwise_dists == -np.inf] = 0.0
    pairwise_dists[pairwise_dists == +np.inf] = 0.0
    pairwise_dists[nan_idx] = 0.0
    return pairwise_dists

class graph_diffusion():
    """
        Weak Supervision Method to classify using spectral graph
        analysis on a transducive setup and parameterized similarity
        measure between two essays.
    """
    def __init__(self,range_min,range_max,similarity_measure,neighbourhood="exponential"):
        """
            Input:
                range_min --> minimum range of marks awarded
                range_max --> maximum range of marks awarded
                similarity_measure --> the kernel that is to be used.
                                       Preferably a one-one mapping.
                neighbourhood --> can be either "exponential" or "average" or "stochastic"
                                  neighbourhood means that the scaling used in diffusion
                                  follows the specified curve and method.
            Output:
                None
            Note:
                "average" <- Takes the average of all the predicted values.
                "exponential" <- A weighted average of the values is taken with weights
                                  as e^-i where i is the i'th iteration of the heat matrix.
                "stochastic" <- The value is chosen among the values with a
                                  probability of e^-i where i is the i'th iteration
                                  of the heat matrix.
        """
        self.range_min = range_min
        self.range_max = range_max
        self.similarity_measure = similarity_measure
        self.neighbourhood = neighbourhood

    def calculate_degree_matrix(self,W):
        """
            Input:
                W --> The similarity measure of the points pairwise taken.
                      Can be interpreted as a graph.
            Output:
                D --> The degree matrix of the graph W.
            Note:
                None
        """
        return np.diag(sum(W.T))

    # graph adjacenvy matrix formulation
    def formulate_graph_laplacian(self,X):
        """
            Input:
                X --> The feature set of essays, of size N * D
            Output:
                L --> Laplacian Matrix of the graph formed by the essay set X.
                      We form the graph W from X by computing the similarity
                      between x and y for all x and y belonging to X
                      Then we find the degree matrix D of W, which is a diagonal
                      matrix.
                      Laplacian L is defined as
                                            L = D - W
                      Normalized Laplacian is defined as
                                        l_norm = D^(-1/2)*L*D^(-1/2)
                      Normalized Laplacian are known to work marginally better
                      than in graph diffusion.
                D -->  The degree matrix D of W
            Note:
                None
        """
        W = self.similarity_measure(X)
        D = self.calculate_degree_matrix(W)
        return csgraph.laplacian(W, normed=True), D

    def train(self,x_train,x_test,y_train):
        """
            Input:
                x_train --> The training samples.
                x_test --> The testing samples.
                y_train --> The ground truth of the training samples.
            Output:
                E_val --> The eigenvalues of the generalized eigen equation
                          of the form
                                    L*X = (lambda)D*X
                          where L is the graph laplacian and D is the
                          Degree Matrix of the graph formed by the points in
                          both the training set and the testing set.
                E_vec --> The eigenvectors in the aformentioned eigen equation.
            Note:
                Y --> n*l(no of categories) matrix. A column has values 1, -1, 0
                      for present, not present and unknown.
                Remark - The formulation is transducive,
                         ie. the training set and the test
                         set is known at the time of training.
                Impl. Remark - scipy.linalg.eigh is used instead of
                               scipy.linalg.eig because D is diagonal matrix
        """
        self.test_size = len(x_test)
        self.train_size = len(y_train)
        self.dim = self.range_max - self.range_min + 1
        self.Y = np.zeros((self.train_size + self.test_size, self.dim))
        rng = [ val for val in xrange(0, self.dim) ]
        for itx in xrange(0, len(y_train)):
            self.Y[itx, rng] = -1
            self.Y[itx,int(y_train[itx])-range_min] = 1 # Subtract to compensate for non zero range start (refer self.predict)
        self.X = np.concatenate((x_train,x_test),0)
        self.L,self.D = self.formulate_graph_laplacian(self.X)
        [self.E_val,self.E_vec_U] = scipy.linalg.eigh(self.L,self.D)

    def exp_neighbourhood(self,Z):
        """
            Input:
                Z --> The heat matrix at different time scales stored row wise.
            Output:
                Weighted average of all the predicted values with weights
                as e^-i where i is the i'th iteration of the heat matrix.
        """
        ans = np.zeros(self.test_size)
        weighted_denom = 0
        for i in xrange(0,itr):
            weighted_denom += scipy.exp(-1*i)
            ans += scipy.exp(-1*i)*Z[:,i]
        return np.round(ans/weighted_denom)

    def average_neighbourhood(self,Z):
        """
            Input:
                Z --> The heat matrix at different time scales stored row wise.
            Output:
                Average of all the predicted values.
        """
        return np.round(sum(Z.T)/itr)

    def stochastic_neighbourhood(self, Z):
        """
            Input:
                Z --> The heat matrix at different time scales stored row wise.
            Output:
                Randomly chosen value from set of predicted values (in a column)
                with probability as e^-i where i is the i'th iteration
                of the heat matrix.
            Note:
                Should converge to "exp_voting" method given enough iterations.
        """
        weights = [scipy.exp(-1*i) for i in xrange(0,self.itr)]
        weights = weights/sum(weights)
        ans = np.zeros(self.test_size)
        for i in xrange(0,self.test_size):
            ans[i] = np.random.choice(Z[i,:], p=weights)
        return ans

    def predict(self):
        """
            Input: (indirect)
                x_test --> The testing samples. However, as the setup is
                           transducive, we need x_test during training.
            Output:
                Voted values of prediction over the various heat matrix of
                different times as defined by
                        t = k*(a^i) with k and a as consts, and i as iteration.
                and heat matrix is defined as
                        H(t) = E_vec*exp(-1*E_val*t)*transpose(E_vec)
                where E_vec and E_val are the eigenvalues & eigenvectors of the
                generalized eigen equation of the form L*X = (lambda)D*X with
                L being the graph laplacian and D being the degree matrix.
                Then the diffused values are computed by
                        Y(t) = H(t)*Y(0)
                where Y(0) is the n*l(no of categories) matrix. A column has values 1, -1, 0
                for present, not present and unknown.
                Find the maximum over a row of Y(t) to find the classification wrt to Y(t).
                Now, use neighbourhood diffusion to find the final classification estimate
                from Y(t) for all t, for the x_test, ie. y_pred.
            Note:
                Maximum voting is used here to predict.
                Small t implies local diffusion.
                Large t implies global diffusion.
        """
        self.itr = 5
        Z = np.zeros((self.test_size,self.itr))
        for i in xrange(0,self.itr):
            t =  0.000000001*(100**i)
            temp = scipy.exp(-self.E_val*t)
            H1 = np.dot(np.dot(self.E_vec_U,np.diag(temp)),self.E_vec_U.T)
            Y1 = np.dot(H1,self.Y)
            Z1 = np.zeros(self.test_size)
            for j in xrange(self.train_size,len(Y1)): # Only the unlabeled (x_test) is labeled
                max_ind = np.argmax(Y1[j,:])
                Z1[j - self.train_size] = max_ind + self.range_min # Sum to compensate for non zero range start (refer self.train)
            Z[:,i] = Z1
        if self.neighbourhood == "exponential":
            return self.exp_neighbourhood(Z)
        elif self.neighbourhood == "average":
            return self.average_neighbourhood(Z)
        elif self.neighbourhood == "stochastic":
            return self.stochastic_neighbourhood(Z)
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
            stat_obj = self.stat_class(range_min=range_min,range_max=range_max, \
                                        similarity_measure=self.similarity_measure, \
                                        neighbourhood="stochastic") # reflection bitches
            stat_obj.train(x_train,x_test,y_train)
            y_pred = np.matrix(stat_obj.predict()).T
            cohen_kappa_rating = own_wp.quadratic_weighted_kappa(y_test,y_pred,\
                                    self.range_min,self.range_max)
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
        #dim_red = LDA()
        #X_train = dim_red.fit_transform(X_train, Y_train)
        diffusion_k_cross = k_fold_cross_validation(cross_valid_k, \
                                                    graph_diffusion, X_train,Y_train, \
                                                    range_min,range_max,gaussian_kernel)
        print diffusion_k_cross.execute()
