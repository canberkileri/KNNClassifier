"""
Author: Canberk İleri <canberkileri@gmail.com>
License: BSD-3-Clause
"""

import os

try:
	import numpy as np
except ImportError:
	c = input( "Trying to Install required module: requests Y/n\n")
  
	if c == "Y" or c == "y":
		os.system('python3 -m pip install numpy') 
        

try:
	import pandas as pd
except ImportError:
	c = input( "Trying to Install required module: requests Y/n\n")
  
	if c == "Y" or c == "y":
		os.system('python3 -m pip install pandas')


import numpy as np
import pandas as pd

def mode(X):    
    """
    Computes the mode of X. The mode of a set of data values is the value that appears most often. 
    
    Note: In the case of equality, this method selects the smallest in numerical data.
    And done alphabetically in data containing characters.
        
    Argument:
        X -- Input value: array, series or list.
    
    Return: -- The mode of X.
    """
    
    if(type(X) == np.ndarray or type(X) == pd.core.series.Series):
        return max(set(X.tolist()), key=X.tolist().count)
    elif type(X) == list:
        return max(set(X), key=X.count)
    else:
        assert False,"You must enter a data type whose mode can be found. Pandas Series, Numpy Array, etc."
        

def random_train_test_split(X, Y, test_size = 0.2, random_state = None):
    """
    Describe: It splits your arrays or matrices into random training and test subsets.
        
    Arguments:
        X -- Features
        
        Y -- Labels
        
        test_size -- It takes a value between 0 and 1.
        
        random_state -- You should use it if you don't want a different partitioning each time. This option is often 
        used to prevent variation during testing. Must be convertible to 32 bit unsigned integers.
    
    Returns:
        train_X, test_X, train_Y, test_Y
    """
       
    assert test_size<1, "It takes a value between 0 and 1."
    
    np.random.seed(random_state)
    rand = np.random.rand(X.shape[0])
    split = rand < np.percentile(rand,(1-test_size)*100)
        
    train_X = X[split]
    train_Y = Y[split]
    test_X =  X[~split]
    test_Y = Y[~split]
    
    return train_X, test_X, train_Y, test_Y
        

def KFold(knn, X, Y, cv = 5, random_state = None):
    """
    Describe: Cross-validation is any of various similar model validation techniques for 
    assessing how the results of a statistical analysis will generalize to an independent data set. 
    It is mainly used in settings where the goal is prediction,and one wants to estimate how accurately
    a predictive model will perform in practice.
           
    Arguments:
        knn -- K-Nearest Neighbour Class
        
        X -- Features
        
        Y -- Labels
        
        cv -- Number of splits.
        
        random_satate -- You should use it if you don't want a different partitioning each time. This option is often 
        used to prevent variation during testing. Must be convertible to 32 bit unsigned integers.
    
    Return: Mean of accuracy.
    
    Notes: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
    """
    
    assert type(knn) == KNNClassifier, "Please use the KNNClassifier you have imported."    
    
    X = X.sample(frac=1, random_state=random_state)
    Y = Y.sample(frac=1, random_state=random_state)
    
    size = X.shape[0]//cv
    acc_list = []
    
    for i in range(cv):
        test_X = X[(i*size):((i+1)*size)]
        test_Y = Y[(i*size):((i+1)*size)]
        train_X = pd.concat([X[:(i*size)], X[((i+1)*size):]])
        train_Y = pd.concat([Y[:(i*size)], Y[((i+1)*size):]])
        knn.fit(train_X,train_Y)
        pred_Y = knn.predict(test_X)   
        acc, conf_matrix = knn.evaluation(test_Y,pred_Y)
        acc_list.append(acc.values)
        
    return np.mean(acc_list)

        
class KNNClassifier():
    """
    Describe: K-Nearest Neighbour Class for Classification
        
    Note: All your data except the target feature must be of int, float or bool type.
    
    Arguments:
        K -- The number of closest neighbors to be considered.
        
        scale --
            - "min-max":
            - "z-score","standardization":
            - "mean-normalization":
                
        distance_metrics -- "minkowski", "manhattan", "euclidean", "canberra", "braycurtis", "chebyshev"
        
        p -- It is a parameter specific to the Minkowski distance metric.
            - p = 1 -> distance metric = manhattan
            - p = 2 -> distance metric = euclidean
            
        weighting -- 
            - uniform: Classic approach of k-nearest neighbour.
            - distance:  Weighted knn approach.
            - Or you can write your own function. For example, you can write a function 
            that does the weighting function according to 1 / d ** 2. 
        
    """
    
    def __init__(self, K =5, scale = "min-max", distance_metric = "minkowski", p = 2, weighting = "uniform"):
        self.K = K
        self.scale = scale
        self.distance_metric = distance_metric
        self.p = p
        self.weighting = weighting
        
        
    def fit(self, train_X, train_Y):
        """
        Describe: It presents your training set to the class.
        
        Notes: All your data except the target feature must be of int, float or bool type.
   
        """      
        assert type(train_X) == pd.core.frame.DataFrame or type(train_X) == np.ndarray, "Please use type of DataFrame or NumPy Array."
        assert pd.api.types.is_numeric_dtype(train_X.all()), "There are non-numerical variables in your data. All your data must be of int, float or bool type."
        assert self.K < train_X.shape[0],"Your training set cannot be less than K. Please redefine the class."
		
        self.train_X = train_X
        self.train_Y = train_Y
        
        if type(self.train_X) == pd.core.frame.DataFrame:        
            self.train_X.index = range(train_X.shape[0])
        
        if type(self.train_Y) == pd.core.frame.DataFrame or type(self.train_Y) == pd.core.series.Series:
            self.train_Y.index = range(train_Y.shape[0])
    
    
    def predict(self, test_X):
        """
        Describe: It looks at your training set that you introduced with the Fit method and predicts the labels of the test set you gave.
        
        Argument: 
           - test_X -- Data set whose labels you want to predict.
                
        Return:
           - pred_Y -- Predicted labels.
            
        Notes: All your data except the target feature must be of int, float or bool type.
    
        """
        
        assert type(test_X) == pd.core.frame.DataFrame or type(self.train_X) == np.ndarray, "Please use DataFrame data type or NumPy Array."
        assert pd.api.types.is_numeric_dtype(test_X.all()), "There are non-numerical variables in your data. All your data must be of int, float or bool type."
        
        if type(test_X) == pd.core.frame.DataFrame: 
            test_X.index = range(test_X.shape[0])
        
        #Epsilon is a very small number to avoid dividing by zero.
        epsilon = 10**-12
        
        
        #Scaling Operations
        if self.scale == "min-max":
            self.train_X = (self.train_X - self.train_X.min(axis=0)) / (self.train_X.max(axis=0) - self.train_X.min(axis=0) + epsilon)
            test_X = (test_X - test_X.min(axis=0)) / (test_X.max(axis=0) - test_X.min(axis=0) + epsilon)
        elif self.scale == "standardization" or self.scale == "z-score":
            self.train_X = (self.train_X - self.train_X.mean(axis=0)) / (self.train_X.std(axis=0) + epsilon)
            test_X = (test_X - test_X.mean(axis=0)) / (test_X.std(axis=0) + epsilon)
        elif self.scale == "mean-normalization":
            self.train_X = (self.train_X - self.train_X.mean(axis=0)) / (self.train_X.max(axis=0) - self.train_X.min(axis=0) + epsilon())
            test_X = (test_X - test_X.mean(axis=0)) / (test_X.max(axis=0) - test_X.min(axis=0) + epsilon)

        #Choice of the function to be applied according to the selected KNN method.
        if self.weighting == "uniform":
            w = lambda X : X
        elif self.weighting == "distance":            
            def w(x):
                weights = np.ones_like(x, dtype=float)
                if any(x == 0):
                    weights[x != 0] = 0
                else:
                    weights = 1/x                                       
                return weights
        elif callable(self.weighting):
            w = self.weighting


        #The Minkowski distance is a metric that is the generalization of both Euclidean and Manhattan distance.
        #If you choose Manhattan, you would use Minkowski as p = 1.
        #If you choose Euclidean, you would use Minkowski as p = 2.
        if  self.distance_metric == "manhattan":
            self.p = 1
            self.distance_metric = "minkowski"
        elif self.distance_metric == "euclidean":
            self.p = 2
            self.distance_metric = "minkowski"    
        
        #Creating the distance matrix.
        #The difference of each training record with the entire test sample table is taken.
        #Then, various operations are applied on it according to the selection.
        distance_matrix = np.zeros((self.train_X.shape[0],test_X.shape[0]))
        
        if type(self.train_X) == pd.core.frame.DataFrame:
            self.train_X = self.train_X.to_numpy()
        if type(test_X) == pd.core.frame.DataFrame:
            test_X = test_X.to_numpy()
                        
        for u in range(self.train_X.shape[0]):
            if self.distance_metric == "minkowski":                 
                distance_matrix[u, :] = np.sum(np.abs(self.train_X[u] - test_X)**self.p,axis=1)**(1/self.p)
            elif self.distance_metric == "canberra":
                distance_matrix[u, :] = np.sum(np.abs(self.train_X[u] - test_X)/(np.abs(self.train_X[u])+np.abs(test_X)),axis=1)
            elif self.distance_metric == "braycurtis":
                distance_matrix[u, :] = np.sum(np.abs(self.train_X[u] - test_X)/(np.sum(np.abs(self.train_X[u]))+np.abs(test_X)),axis=1)
            elif self.distance_metric == "chebyshev":
                distance_matrix[u, :] = np.max(np.abs(self.train_X[u] - test_X),axis=1)
            else:
                assert False,"The metric you entered does not exist. Please read the method description."
        
        #Creating and filling matrices to keep distances and indexes information.
        sorted_distance_index = np.zeros((test_X.shape[0],self.K), dtype=int)
        sorted_distance_dist = np.zeros((test_X.shape[0],self.K))
        for z in range(distance_matrix.shape[1]):
            
            dist_argsort = distance_matrix.T[z].argsort()[:self.K]
            sorted_distance_index[z] = dist_argsort
            sorted_distance_dist[z] = w(distance_matrix.T[z][dist_argsort])
                       
              
        pred_Y = np.zeros((test_X.shape[0])) 
        if type(self.train_Y[0]) == np.str_ or type(self.train_Y[0]) == str:
            pred_Y = pred_Y.astype('str')      
        
        
        if self.weighting == "uniform":
            for b in range(test_X.shape[0]):
                #Fetch tags with index information of sorted neighbors. And computes the mode of the array.
                pred_Y[b] =  mode(self.train_Y[sorted_distance_index[b]])

        else:
            if type(self.train_Y) == np.ndarray:
                self.train_Y = pd.Series(self.train_Y)
            sum_of_classes = []
            for k in range(test_X.shape[0]):
                #labels
                aggry = self.train_Y[sorted_distance_index[k]]
                #distances
                aggrx = sorted_distance_dist[k]
                #names of classes
                classes = np.sort(aggry.value_counts().index.values)
               
                sum_of_classes.clear()
                #Total of distances for each class.
                for a in classes:                   
                    sum_of_classes.append(np.sum(aggrx[aggry == a]))
                               
                sum_of_classes_sum = sum(sum_of_classes)
                
                ratio = sum_of_classes / sum_of_classes_sum
                
                #Index information of values sorted from small to large.
                argsort_ratio = ratio.argsort()
                
                #Taking the label of the biggest one.
                pred_class = classes[argsort_ratio[-1]]
                                             
                pred_Y[k] = pred_class
        
        return pred_Y
    @staticmethod
    def evaluation(true_Y, pred_Y):
        """
        Describe: It compares the class labels you predicted with the correct labels.
        
        Arguments:
            - true_Y: Correct labels.
            - pred_Y: Predicted labels.           
        
        Returns:
            - acc: Accuracy value of model.-> true values / all
            - df_confusion: Confusion Matrix
    
        """
        
        assert ((type(true_Y) == np.ndarray or
                type(true_Y) == pd.core.series.Series)
        or (type(pred_Y) == np.ndarray or
                type(pred_Y) == pd.core.series.Series)), "You must enter Pandas Series, Numpy Array, etc."
        
        
        if type(true_Y) == np.ndarray:
            true_Y = pd.Series(true_Y)
        if type(pred_Y) == np.ndarray:
            pred_Y = pd.Series(pred_Y)
        
        true_Y.index = range(true_Y.shape[0])
        
        count = (true_Y == pred_Y).value_counts()
        
        acc = count[count.index == True] / count.sum()
        
        df_confusion = pd.crosstab(true_Y, pred_Y, rownames=['Actual'], colnames=['Predicted'], margins=True)
        
        return acc, df_confusion    


    @staticmethod
    def classification_report(conf_matrix , only_print=True):
        """
        Describe: It calculates precision, recall and f1 values of classes.            
        
        Arguments:
            conf_matrix -- Confusion matrix as DataFrame.
            only_print --
                - True: It prints precision, recall and f1 values of classes but it do not returns any value.
                - False: It prints values and also returns a dataframe containing precision, recall and f1 values of classes.
        
        Return: Returns a dataframe containing precision, recall and f1 values of classes.
    
        """
        
        assert type(conf_matrix) == pd.core.frame.DataFrame, "Please use DataFrame data type."
        
        
        all_information = "   precision   recall   f1-score\n\n"
        conf_matrix_raw = conf_matrix.iloc[:-1,:-1].values
        n_class = conf_matrix_raw.diagonal().shape[0]
        precision_list, recall_list, f1_list = [],[],[]
        for i in range(n_class):
            #Numpy warning has been blocked from printing.
            np.seterr(all='print')
            
            precision = conf_matrix_raw.diagonal()[i] / conf_matrix_raw.sum(axis=0)[i]
            recall = conf_matrix_raw.diagonal()[i] / conf_matrix_raw.sum(axis=1)[i]            
            
            precision = np.nan_to_num(precision)
            recall = np.nan_to_num(precision)
            
            f1 = (2*precision*recall) / (precision+recall)
            
            f1 = np.nan_to_num(f1)
                       
            recall_list.append(recall)
            precision_list.append(precision)
        
            f1_list.append(f1)
            name = conf_matrix.index[:-1].values[i]      
            all_information+= (str(name) + "    " +
                               str(round(precision,4)) + "    " +
                                   str(round(recall,4)) + "    " +
                                       str(round(f1,4)) +"\n")
        print(all_information)
        if only_print == False:
            info_df = pd.DataFrame([precision_list, recall_list, f1_list], index=["precision","recall","f1"], columns=conf_matrix.index[:-1].values)
            info_df.replace(np.nan, 0, inplace = True)
            return info_df.T
