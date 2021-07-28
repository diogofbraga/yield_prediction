#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kernel modules.
"""
import pandas as pd
import numpy as np
import csv # Added by diogofbraga

import grakel.kernels as kernels
import sklearn.metrics.pairwise as sklearn_kernels

#from tools.machine_learning.grakel_nonlinear.vertex_histogram import VertexHistogram
#from tools.machine_learning.grakel_nonlinear.weisfeiler_lehman import WeisfeilerLehman
from grakel.kernels.weisfeiler_lehman import WeisfeilerLehman
from grakel.kernels.vertex_histogram import VertexHistogram

class kernel():
    """A class that defines and calculates kernels using GraKel."""
    
    def __init__(self, kernel_name, base_kernel=None):
        self.kernel_name = kernel_name
        self.base_kernel = base_kernel
        
    def define_kernel(self, *args, **kwargs):
        """
        Defines the graph kernel.
        Parameters
        ----------
        *args :
            Graph kernel parameters.
        **kwargs :
            Graph kernel parameters.
        Returns
        -------
        None.
        """
        base_kernel = self.base_kernel
        if base_kernel is None:
            base_kernel = 'VertexHistogram'
        k = getattr(kernels, self.kernel_name)
        k_base = getattr(kernels, base_kernel)
        #self.kernel = k(base_kernel=k_base, *args, **kwargs)
        self.kernel = k(base_graph_kernel=k_base, *args, **kwargs)
        
    def fit_and_transform(self, X):
        """
        Fit and transform on the same dataset. Calculates X_fit by X_fit 
        kernel matrix.
        """
        self.fitted_kernel = self.kernel.fit_transform(X)
    
    def transform_data(self, X):
        """
        Calculates X_fit by X_transform kernel matrix.
        """
        self.transformed_kernel = self.kernel.transform(X)
    
    def calcualte_reduced_X(self, X):
        """
        
        """
        missing_mol_indices = []
        present_mol_indices = []
        reduced_X = []
         
        # Get indices where mols are missing and create new list with no 
        # missing mols.
        for i, x in enumerate(X):
            if pd.isnull(x):
                 missing_mol_indices.append(i)
            else:
                present_mol_indices.append(i)
                reduced_X.append(x)
                
        present_mol_indices = np.array(present_mol_indices)             
        
        return reduced_X, missing_mol_indices, present_mol_indices
        
    def calculate_kernel_matrices(self, X_train, X_test, **kernel_params):
        """
        Fit and transform the X_train data. Calculate the kernel matrix between
        the fitted data (X_train) and X_test.
        Parameters
        ----------
        X_train : Series, list, numpy array 
            Training set of molecular graphs. Input must be iterable.
        X_test : Series, list, numpy array 
            Test set of molecular graphs. Input must be iterable.
        Returns
        -------
        k_train : numpy array
            The kernel matrix between all pairs of graphs in X_train.
        k_test : TYPE
            The kernel matrix between all pairs of graphs in X_train and 
            X_test.
        """
        self.define_kernel(normalize=True, **kernel_params)
        
        self.fit_and_transform(X_train)
        if X_test is not None:
            self.transform_data(X_test)
    
        k_train = self.fitted_kernel
        if X_test is not None:
            k_test = self.transformed_kernel
            return k_train, k_test   
        else:
            return k_train
    
    def calculate_kernel_matrices_with_missing_mols(self, X_train, X_test, **kernel_params):
        """
        Fit and transform the X_train data. Calculate the kernel matrix between
        the fitted data (X_train) and X_test.
        Parameters
        ----------
        X_train : Series, list, numpy array 
            Training set of molecular graphs. Input must be iterable.
        X_test : Series, list, numpy array 
            Test set of molecular graphs. Input must be iterable.
        Returns
        -------
        k_train : numpy array
            The kernel matrix between all pairs of graphs in X_train.
        k_test : TYPE
            The kernel matrix between all pairs of graphs in X_train and 
            X_test.
        """
        self.define_kernel(normalize=True, **kernel_params)
        
        if X_train.isnull().values.any():
            print('nan in X_train')
            reduced_X_train, \
                missing_mol_indices_X_train, \
                    present_mol_indices_X_train \
                        = self.calcualte_reduced_X(X_train)
            
            len_X_train = len(X_train)
            len_reduced_X_train = len(reduced_X_train)
        
            # Calculate kernel matrix on non-missing molecules
            reduced_k_train = self.kernel.fit_transform(reduced_X_train)
            
            # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
            np.add(
                reduced_k_train, 
                np.ones((len_reduced_X_train, len_reduced_X_train)), 
                reduced_k_train
                ) 
        
            # missing molecules have value 1 or 2, initialise with ones
            k_train = np.ones((len_X_train, len_X_train)) 
            
            reduced_index_X_train = present_mol_indices_X_train[
                np.arange(reduced_k_train.shape[0])
                ]
        
            for i in range(len_reduced_X_train):
                k_train[reduced_index_X_train[i], reduced_index_X_train] \
                    = reduced_k_train[i, :]
        
            for i in missing_mol_indices_X_train:
                for j in missing_mol_indices_X_train:
                    k_train[i, j] = 2
                
        else:
            print('no nan in X_train')            
            k_train = self.kernel.fit_transform(X_train)
            
            len_X_train = len(X_train)
            
            # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
            np.add(
                k_train, 
                np.ones((len_X_train, len_X_train)), 
                k_train
                ) 
                        
            len_reduced_X_train = len(X_train)
            reduced_index_X_train = np.arange(k_train.shape[0])
            missing_mol_indices_X_train = []
        
        if X_test.isnull().values.any():
            print('nan in X_test')
            reduced_X_test, \
                missing_mol_indices_X_test, \
                    present_mol_indices_X_test \
                        = self.calcualte_reduced_X(X_test)
            
            len_X_test = len(X_test)
            len_reduced_X_test = len(reduced_X_test)
            
            # Calculate kernel matrix on non-missing molecules
            reduced_k_test = self.kernel.transform(reduced_X_test)
            
            # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
            np.add(
                reduced_k_test, 
                np.ones((len_reduced_X_test, len_reduced_X_train)), 
                reduced_k_test
                ) 
        
            # missing molecules have value 1 or 2, initialise with ones
            k_test = np.ones((len_X_test, len_X_train)) 
            
            reduced_index_X_test = present_mol_indices_X_test[
                np.arange(reduced_k_test.shape[0])
                ]
        
            for i in range(len_reduced_X_test):
                k_test[reduced_index_X_test[i], reduced_index_X_train] \
                    = reduced_k_test[i, :]
        
            for i in missing_mol_indices_X_test:
                for j in missing_mol_indices_X_train:
                    k_test[i, j] = 2
                        
        else:
            print('no nan in X_test')
            reduced_k_test = self.kernel.transform(X_test)
            
            len_X_test = len(X_test)
            
            np.add(
                reduced_k_test, 
                np.ones((len_X_test, len_reduced_X_train)), 
                reduced_k_test
                )
            
            k_test = np.ones((len_X_test, len_X_train)) 
            
            reduced_index_X_test = np.arange(k_test.shape[0])
            
            for i in range(len_X_test):
                k_test[reduced_index_X_test[i], reduced_index_X_train] \
                    = reduced_k_test[i, :]
    
        return k_train, k_test  
    
        

    
    def non_linearity(self, K, kernel_function):

        print("Kernel function:", kernel_function)

        if kernel_function is 'linear':
            pass
        
        elif kernel_function is 'polynomial':
            self.scale = 1
            self.bias = 0
            self.degree = 3
            K = (self.scale * K + self.bias) ** self.degree
        
        elif kernel_function is 'sigmoidlogistic': # Normalised matrix is returning values bigger than 1
            self.scale = 0.01
            K = 1 / (1 + np.exp(-K * self.scale))

        elif kernel_function is 'sigmoidhyperbolictangent': # Normalised matrix is returning values bigger than 1
            self.scale = 0.001
            self.bias = 0
            K = np.tanh(self.scale * K + self.bias)

        elif kernel_function is 'sigmoidarctangent':
            self.scale = 0.01
            self.bias = 0
            K = np.arctan(self.scale * K + self.bias)

        elif kernel_function is 'gaussian':
            sigma = float(1/K.shape[1])
            variance = np.power(sigma,2)
            K = np.exp(-((np.abs(K)) ** 2)/(2*variance))

        elif kernel_function is 'exponential':
            sigma = float(1/K.shape[1])
            variance = np.power(sigma,2)
            K = np.exp(-(np.abs(K))/(2*variance))
        
        elif kernel_function is 'rbf':
            self.gamma = float(1/K.shape[1])
            K = np.exp(-self.gamma * (np.abs(K)) ** 2)
        
        elif kernel_function is 'laplacian':
            standard_deviation = float(1/self.km_test.shape[1])
            K = np.exp(-(np.abs(K))/standard_deviation)
        
        elif kernel_function is 'rationalquadratic': # Return a matrix with only 0s
            standard_deviation = float(1/K.shape[1])
            bias = 0
            K = 1 - (((np.abs(D)) ** 2)/((np.abs(K)) ** 2) + bias)

        elif kernel_function is 'multiquadratic':
            bias = 1
            K = np.sqrt(((np.abs(K)) ** 2) + np.power(bias,2))

        elif kernel_function is 'inversemultiquadratic':
            bias = 1
            K = 1 / np.sqrt(((np.abs(K)) ** 2) + np.power(bias,2))
        
        elif kernel_function is 'power': # Problems with the division
            degree = 2
            K = -(np.abs(K) ** degree)

        elif kernel_function is 'log': # Problems with the division
            degree = 2
            K = -np.log((np.abs(K) ** degree) + 1)

        elif kernel_function is 'cauchy':
            sigma = float(1/K.shape[1])
            variance = np.power(sigma,2)
            K = 1 / (1 + ((np.abs(K)) ** 2)/variance)

        return K


    def multiple_descriptor_types(self, X_train, X_test, **kernel_params):
        k_train = 1

        if 'kernel_function' in kernel_params:
            kernel_function = kernel_params.pop('kernel_function', None)
        
        if X_test is not None:
            k_test = 1
            
            if X_train.isnull().values.any() or X_test.isnull().values.any(): 
                for i in X_train:
                    train, test = self.calculate_kernel_matrices_with_missing_mols(
                        X_train[i], X_test[i], **kernel_params
                        )

                    train = self.non_linearity(train, kernel_function)
                    test = self.non_linearity(test, kernel_function)

                    k_train = k_train * train
                    k_test = k_test * test
            else:
                for i in X_train:
                    train, test = self.calculate_kernel_matrices(
                        X_train[i], X_test[i], **kernel_params
                        )

                    train = self.non_linearity(train, kernel_function)
                    test = self.non_linearity(test, kernel_function)

                    k_train = k_train * train
                    k_test = k_test * test
                    
        else:
            k_test = None
            for i in X_train:
                train = self.calculate_kernel_matrices(
                    X_train[i], None, **kernel_params
                    )

                train = self.non_linearity(train, kernel_function)

                k_train = k_train * train          
            
        return k_train, k_test
    
class sklearn_kernel():
    """A class that defines and calculates kernels using Sklearn."""
    
    def __init__(self, kernel_name):
        self.kernel_name = kernel_name
        
    # def define_kernel(self):
    #     """
    #     Defines the kernel.

    #     Parameters
    #     ----------
    #     None.
        
    #     Returns
    #     -------
    #     None.

    #     """.
        k = getattr(sklearn_kernels, self.kernel_name)
        self.kernel = k
    
    def calcualte_reduced_X(self, X):
        """
        
        """
        missing_mol_indices = []
        present_mol_indices = []
        reduced_X = []
         
        # Get indices where mols are missing and create new list with no 
        # missing mols.
        for i, x in X.iterrows():
            if x.isnull().values.any():
                 missing_mol_indices.append(X.index.get_loc(i))
            else:
                present_mol_indices.append(X.index.get_loc(i))
                reduced_X.append(x.values)
                
        present_mol_indices = np.array(present_mol_indices)             
        
        return reduced_X, missing_mol_indices, present_mol_indices
        
    def calculate_kernel_matrices(self, X_train, X_test):
        """
        Fit and transform the X_train data. Calculate the kernel matrix between
        the fitted data (X_train) and X_test.

        Parameters
        ----------
        X_train : Series, list, numpy array 
            Training set. Input must be iterable.
        X_test : Series, list, numpy array 
            Test set. Input must be iterable.

        Returns
        -------
        k_train : numpy array
            The kernel matrix between all pairs in X_train.
        k_test : TYPE
            The kernel matrix between all pairs in X_train and 
            X_test.

        """
    
        k_train = self.kernel(X_train)
        k_test = self.kernel(X_train, X_test)
        return k_train, k_test   
    
    def calculate_kernel_matrices_with_missing_mols(self, X_train, X_test):
        """
        Fit and transform the X_train data. Calculate the kernel matrix between
        the fitted data (X_train) and X_test.

        Parameters
        ----------
        X_train : Series, list, numpy array 
            Training set. Input must be iterable.
        X_test : Series, list, numpy array 
            Test set of. Input must be iterable.

        Returns
        -------
        k_train : numpy array
            The kernel matrix between all pairs in X_train.
        k_test : TYPE
            The kernel matrix between all pairs in X_train and 
            X_test.

        """
        # self.define_kernel(normalize=True, **kernel_params)
        
        if X_train.isnull().values.any():
            print('nan in X_train')
            reduced_X_train, \
                missing_mol_indices_X_train, \
                    present_mol_indices_X_train \
                        = self.calcualte_reduced_X(X_train)
            
            len_X_train = len(X_train)
            len_reduced_X_train = len(reduced_X_train)
        
            # Calculate kernel matrix on non-missing molecules
            reduced_k_train = self.kernel(reduced_X_train)
            
            # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
            np.add(
                reduced_k_train, 
                np.ones((len_reduced_X_train, len_reduced_X_train)), 
                reduced_k_train
                ) 
        
            # missing molecules have value 1 or 2, initialise with ones
            k_train = np.ones((len_X_train, len_X_train)) 
            
            reduced_index_X_train = present_mol_indices_X_train[
                np.arange(reduced_k_train.shape[0])
                ]
        
            for i in range(len_reduced_X_train):
                k_train[reduced_index_X_train[i], reduced_index_X_train] \
                    = reduced_k_train[i, :]
        
            for i in missing_mol_indices_X_train:
                for j in missing_mol_indices_X_train:
                    k_train[i, j] = 2
                
        else:
            print('no nan in X_train')            
            k_train = self.kernel(X_train)
            
            len_X_train = len(X_train)
            
            # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
            np.add(
                k_train, 
                np.ones((len_X_train, len_X_train)), 
                k_train
                ) 
            
            reduced_X_train = X_train
            len_reduced_X_train = len(X_train)
            reduced_index_X_train = np.arange(k_train.shape[0])
            missing_mol_indices_X_train = []
        
        if X_test.isnull().values.any():
            print('nan in X_test')
            reduced_X_test, \
                missing_mol_indices_X_test, \
                    present_mol_indices_X_test \
                        = self.calcualte_reduced_X(X_test)
            
            len_X_test = len(X_test)
            len_reduced_X_test = len(reduced_X_test)
            
            # Calculate kernel matrix on non-missing molecules
            reduced_k_test = self.kernel(reduced_X_test, reduced_X_train)
            
            # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
            np.add(
                reduced_k_test, 
                np.ones((len_reduced_X_test, len_reduced_X_train)), 
                reduced_k_test
                ) 
        
            # missing molecules have value 1 or 2, initialise with ones
            k_test = np.ones((len_X_test, len_X_train)) 
            
            reduced_index_X_test = present_mol_indices_X_test[
                np.arange(reduced_k_test.shape[0])
                ]
        
            for i in range(len_reduced_X_test):
                k_test[reduced_index_X_test[i], reduced_index_X_train] \
                    = reduced_k_test[i, :]
        
            for i in missing_mol_indices_X_test:
                for j in missing_mol_indices_X_train:
                    k_test[i, j] = 2
                        
        else:
            print('no nan in X_test')
            reduced_k_test = self.kernel(X_test, reduced_X_train)
            
            len_X_test = len(X_test)
            
            np.add(
                reduced_k_test, 
                np.ones((len_X_test, len_reduced_X_train)), 
                reduced_k_test
                )
            
            k_test = np.ones((len_X_test, len_X_train)) 
            
            reduced_index_X_test = np.arange(k_test.shape[0])
            
            for i in range(len_X_test):
                k_test[reduced_index_X_test[i], reduced_index_X_train] \
                    = reduced_k_test[i, :]
    
        return k_train, k_test   
    
