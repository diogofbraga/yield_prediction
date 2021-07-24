"""The weisfeiler lehman kernel :cite:`shervashidze2011weisfeiler`."""
# Author: Ioannis Siglidis <y.siglidis@gmail.com>
# License: BSD 3 clause
import collections
import warnings

import numpy as np
import joblib
import csv

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from grakel.graph import Graph
from grakel.kernels import Kernel
#from tools.machine_learning.grakel_nonlinear.vertex_histogram import VertexHistogram
from grakel.kernels.vertex_histogram import VertexHistogram

# Python 2/3 cross-compatibility import
from six import iteritems
from six import itervalues


class WeisfeilerLehman(Kernel):
    """Compute the Weisfeiler Lehman Kernel.
     See :cite:`shervashidze2011weisfeiler`.
    Parameters
    ----------
    n_iter : int, default=5
        The number of iterations.
    base_graph_kernel : `grakel.kernels.Kernel` or tuple, default=None
        If tuple it must consist of a valid kernel object and a
        dictionary of parameters. General parameters concerning
        normalization, concurrency, .. will be ignored, and the
        ones of given on `__init__` will be passed in case it is needed.
        Default `base_graph_kernel` is `VertexHistogram`.
    Attributes
    ----------
    X : dict
     Holds a dictionary of fitted subkernel modules for all levels.
    _nx : number
        Holds the number of inputs.
    _n_iter : int
        Holds the number, of iterations.
    _base_graph_kernel : function
        A void function that initializes a base kernel object.
    _inv_labels : dict
        An inverse dictionary, used for relabeling on each iteration.
    """

    _graph_format = "dictionary"

    def __init__(self, n_jobs=None, verbose=False,
                 normalize=False, n_iter=5, base_graph_kernel=VertexHistogram):
        """Initialise a `weisfeiler_lehman` kernel."""
        super(WeisfeilerLehman, self).__init__(
            n_jobs=n_jobs, verbose=verbose, normalize=normalize)

        self.n_iter = n_iter
        self.base_graph_kernel = base_graph_kernel
        self._initialized.update({"n_iter": False, "base_graph_kernel": False})
        self._base_graph_kernel = None

    def initialize(self):
        """Initialize all transformer arguments, needing initialization."""
        super(WeisfeilerLehman, self).initialize()
        if not self._initialized["base_graph_kernel"]:
            base_graph_kernel = self.base_graph_kernel
            if base_graph_kernel is None:
                base_graph_kernel, params = VertexHistogram, dict()
            elif type(base_graph_kernel) is type and issubclass(base_graph_kernel, Kernel):
                params = dict()
            else:
                try:
                    base_graph_kernel, params = base_graph_kernel
                except Exception:
                    raise TypeError('Base kernel was not formulated in '
                                    'the correct way. '
                                    'Check documentation.')

                if not (type(base_graph_kernel) is type and
                        issubclass(base_graph_kernel, Kernel)):
                    raise TypeError('The first argument must be a valid '
                                    'grakel.kernel.kernel Object')
                if type(params) is not dict:
                    raise ValueError('If the second argument of base '
                                     'kernel exists, it must be a diction'
                                     'ary between parameters names and '
                                     'values')
                params.pop("normalize", None)

            params["normalize"] = False
            params["verbose"] = self.verbose
            params["n_jobs"] = None
            self._base_graph_kernel = base_graph_kernel
            self._params = params
            self._initialized["base_graph_kernel"] = True

        if not self._initialized["n_iter"]:
            if type(self.n_iter) is not int or self.n_iter <= 0:
                raise TypeError("'n_iter' must be a positive integer")
            self._n_iter = self.n_iter # + 1 removed
            self._initialized["n_iter"] = True

    def parse_input(self, X):
        """Parse input for weisfeiler lehman.
        Parameters
        ----------
        X : iterable
            For the input to pass the test, we must have:
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that correspond to the given
            graph format). A valid input also consists of graph type objects.
        Returns
        -------
        base_graph_kernel : object
        Returns base_graph_kernel.
        """
        if self._method_calling not in [1, 2]:
            raise ValueError('method call must be called either from fit ' +
                             'or fit-transform')
        elif hasattr(self, '_X_diag'):
            # Clean _X_diag value
            delattr(self, '_X_diag')

        # Input validation and parsing
        if not isinstance(X, collections.Iterable):
            raise TypeError('input must be an iterable\n')
        else:
            nx = 0
            Gs_ed, L, distinct_values, extras = dict(), dict(), set(), dict()
            for (idx, x) in enumerate(iter(X)):
                is_iter = isinstance(x, collections.Iterable)
                if is_iter:
                    x = list(x)
                if is_iter and (len(x) == 0 or len(x) >= 2):
                    if len(x) == 0:
                        warnings.warn('Ignoring empty element on index: '
                                      + str(idx))
                        continue
                    else:
                        if len(x) > 2:
                            extra = tuple()
                            if len(x) > 3:
                                extra = tuple(x[3:])
                            x = Graph(x[0], x[1], x[2], graph_format=self._graph_format)
                            extra = (x.get_labels(purpose=self._graph_format,
                                                  label_type="edge", return_none=True), ) + extra
                        else:
                            x = Graph(x[0], x[1], {}, graph_format=self._graph_format)
                            extra = tuple()

                elif type(x) is Graph:
                    x.desired_format(self._graph_format)
                    el = x.get_labels(purpose=self._graph_format, label_type="edge", return_none=True)
                    if el is None:
                        extra = tuple()
                    else:
                        extra = (el, )

                else:
                    raise TypeError('each element of X must be either a ' +
                                    'graph object or a list with at least ' +
                                    'a graph like object and node labels ' +
                                    'dict \n')
                Gs_ed[nx] = x.get_edge_dictionary()
                L[nx] = x.get_labels(purpose="dictionary")
                extras[nx] = extra
                distinct_values |= set(itervalues(L[nx]))
                nx += 1
            if nx == 0:
                raise ValueError('parsed input is empty')

        # Save the number of "fitted" graphs.
        self._nx = nx

        # get all the distinct values of current labels
        WL_labels_inverse = dict()

        # assign a number to each label
        label_count = 0
        for dv in sorted(list(distinct_values)):
            WL_labels_inverse[dv] = label_count
            label_count += 1

        # Initalize an inverse dictionary of labels for all iterations
        self._inv_labels = dict()
        self._inv_labels[0] = WL_labels_inverse

        def generate_graphs(label_count, WL_labels_inverse):
            new_graphs = list()
            for j in range(nx):
                new_labels = dict()
                for k in L[j].keys():
                    new_labels[k] = WL_labels_inverse[L[j][k]]
                L[j] = new_labels
                # add new labels
                new_graphs.append((Gs_ed[j], new_labels) + extras[j])
            yield new_graphs

            for i in range(1, self._n_iter):
                label_set, WL_labels_inverse, L_temp = set(), dict(), dict()
                for j in range(nx):
                    # Find unique labels and sort
                    # them for both graphs
                    # Keep for each node the temporary
                    L_temp[j] = dict()
                    for v in Gs_ed[j].keys():
                        credential = str(L[j][v]) + "," + \
                            str(sorted([L[j][n] for n in Gs_ed[j][v].keys()]))
                        L_temp[j][v] = credential
                        label_set.add(credential)

                label_list = sorted(list(label_set))
                for dv in label_list:
                    WL_labels_inverse[dv] = label_count
                    label_count += 1

                # Recalculate labels
                new_graphs = list()
                for j in range(nx):
                    new_labels = dict()
                    for k in L_temp[j].keys():
                        new_labels[k] = WL_labels_inverse[L_temp[j][k]]
                    L[j] = new_labels
                    # relabel
                    new_graphs.append((Gs_ed[j], new_labels) + extras[j])
                self._inv_labels[i] = WL_labels_inverse
                yield new_graphs

        print("----- NUMBER OF ITERATIONS -----:", self._n_iter)
        base_graph_kernel = {i: self._base_graph_kernel(**self._params) for i in range(self._n_iter)}
        if self._parallel is None:
            if self._method_calling == 1:
                for (i, g) in enumerate(generate_graphs(label_count, WL_labels_inverse)):
                    base_graph_kernel[i].fit(g)
            elif self._method_calling == 2:
                K = np.sum((base_graph_kernel[i].fit_transform(g) for (i, g) in
                           enumerate(generate_graphs(label_count, WL_labels_inverse))), axis=0)

        else:
            if self._method_calling == 1:
                self._parallel(joblib.delayed(efit)(base_graph_kernel[i], g)
                               for (i, g) in enumerate(generate_graphs(label_count, WL_labels_inverse)))
            elif self._method_calling == 2:
                K = np.sum(self._parallel(joblib.delayed(efit_transform)(base_graph_kernel[i], g)
                           for (i, g) in enumerate(generate_graphs(label_count, WL_labels_inverse))),
                           axis=0)

        if self._method_calling == 1:
            return base_graph_kernel
        elif self._method_calling == 2:
            return K, base_graph_kernel

    def calculate_distance_kernel(self, mode):
        if mode == 'fit_transform':
            D = np.zeros(self.km_train.shape)

            for row in range(len(self.km_train)): # rows
                for column in range(len(self.km_train[row])): # columns compared with rows
                    D[row][column] = self.km_train[row][row] + self.km_train[column][column] - (2 * self.km_train[row][column])
        else:
            D = np.zeros(self.km_test.shape)

            for row in range(len(self.km_test)): # rows
                for column in range(len(self.km_test[row])): # columns compared with rows
                    D[row][column] = self.km_test[row][row] + self.km_train[column][column] - (2 * self.km_test[row][column])

        return D

    def non_linearity(self, kernel_function, mode):

        #print("----- SUM -----")
        #if mode == 'fit_transform':
        #    print("Kernel matrix before non-linearity: \n", self.km_train)
        #else:
        #    print("Kernel matrix before non-linearity: \n", self.km_test)
        #kernel_function = 'polynomial'
        print("Kernel function:", kernel_function)

        if kernel_function is 'linear':
            if mode == 'fit_transform':
                K = self.km_train
            else:
                K = self.km_test
        
        if kernel_function is 'polynomial':
            scale = 1
            bias = 0
            degree = 2
            if mode == 'fit_transform':
                K = (scale * self.km_train + bias) ** degree
            else:
                K = (scale * self.km_test + bias) ** degree
        
        elif kernel_function is 'sigmoidlogistic': # The matrix values are too small, with 1 we lose the differences
            if mode == 'fit_transform':
                K = 1 / (1 + np.exp(-self.km_train))
            else:
                K = 1 / (1 + np.exp(-self.km_test))

        elif kernel_function is 'sigmoidhyperbolictangent': # Return a matrix with only 1s
            scale = 1
            bias = 0
            if mode == 'fit_transform':
                K = np.tanh(scale * self.km_train + bias)
                #K = np.tan(scale * K + bias) # It doesn't seem to work with the normalize = True
                #K = np.arctan(scale * K + bias) # It doesn't seem to work with the normalize = True
            else:
                K = np.tanh(scale * self.km_test + bias)

        elif kernel_function is 'gaussian':
            if mode == 'fit_transform':
                sigma = float(1/self.km_train.shape[1])
            else:
                sigma = float(1/self.km_test.shape[1])

            D = self.calculate_distance_kernel(mode)
            variance = np.power(sigma,2)
            K = np.exp(-((np.abs(D)) ** 2)/(2*variance))

        elif kernel_function is 'exponential':
            if mode == 'fit_transform':
                sigma = float(1/self.km_train.shape[1])
            else:
                sigma = float(1/self.km_test.shape[1])

            D = self.calculate_distance_kernel(mode)
            variance = np.power(sigma,2)
            K = np.exp(-(np.abs(D))/(2*variance))
        
        if kernel_function is 'rbf':
            if mode == 'fit_transform':
                gamma = float(1/self.km_train.shape[1])
            else:
                gamma = float(1/self.km_test.shape[1])

            D = self.calculate_distance_kernel(mode)
            K = np.exp(-gamma * (np.abs(D)) ** 2)
        
        elif kernel_function is 'laplacian':
            if mode == 'fit_transform':
                standard_deviation = float(1/self.km_train.shape[1])
            else:
                standard_deviation = float(1/self.km_test.shape[1])

            D = self.calculate_distance_kernel(mode)
            K = np.exp(-(np.abs(D))/standard_deviation)
        
        elif kernel_function is 'rationalquadratic': # Return a matrix with only 0s
            if mode == 'fit_transform':
                standard_deviation = float(1/self.km_train.shape[1])
            else:
                standard_deviation = float(1/self.km_test.shape[1])

            D = self.calculate_distance_kernel(mode)
            bias = 0
            K = 1 - (((np.abs(D)) ** 2)/((np.abs(D)) ** 2) + bias)

        elif kernel_function is 'multiquadratic':
            D = self.calculate_distance_kernel(mode)
            bias = 1
            K = np.sqrt(((np.abs(D)) ** 2) + np.power(bias,2))

        elif kernel_function is 'inversemultiquadratic':
            D = self.calculate_distance_kernel(mode)
            bias = 1
            K = 1 / np.sqrt(((np.abs(D)) ** 2) + np.power(bias,2))
        
        elif kernel_function is 'power': # Problems with the division
            D = self.calculate_distance_kernel(mode)
            degree = 2
            K = -(np.abs(D) ** degree)

        elif kernel_function is 'log': # Problems with the division
            D = self.calculate_distance_kernel(mode)
            degree = 2
            K = -np.log((np.abs(D) ** degree) + 1)

        elif kernel_function is 'cauchy':
            if mode == 'fit_transform':
                sigma = float(1/self.km_train.shape[1])
            else:
                sigma = float(1/self.km_test.shape[1])

            D = self.calculate_distance_kernel(mode)
            variance = np.power(sigma,2)
            K = 1 / (1 + ((np.abs(D)) ** 2)/variance)

        #print("Kernel matrix after non-linearity: \n", K)
        return K


    def fit_transform(self, X, kernel_function, y=None):
        """Fit and transform, on the same dataset.
        Parameters
        ----------
        X : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format). If None the kernel matrix is calculated upon fit data.
            The test samples.
        y : Object, default=None
            Ignored argument, added for the pipeline.
        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features
        """
        self._method_calling = 2
        self._is_transformed = False
        self.initialize()
        if X is None:
            raise ValueError('transform input cannot be None')
        else:
            #print("X", X)
            self.km_train, self.X = self.parse_input(X)

        mode = 'fit_transform'
        km = self.non_linearity(kernel_function, mode)

        self._X_diag = np.diagonal(km)
        #print("Xdiag", self._X_diag)
        #print("Xdiag shape", self._X_diag.shape)
        if self.normalize:
            old_settings = np.seterr(divide='ignore')
            #print("Divide km by sqrt(outer result) \n", np.divide(km, np.sqrt(np.outer(self._X_diag, self._X_diag))))
            km = np.nan_to_num(np.divide(km, np.sqrt(np.outer(self._X_diag, self._X_diag))))
            np.seterr(**old_settings)
        return km

    def transform(self, X, kernel_function):
        """Calculate the kernel matrix, between given and fitted dataset.
        Parameters
        ----------
        X : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format). If None the kernel matrix is calculated upon fit data.
            The test samples.
        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features
        """
        self._method_calling = 3
        # Check is fit had been called
        check_is_fitted(self, ['X', '_nx', '_inv_labels'])

        # Input validation and parsing
        if X is None:
            raise ValueError('transform input cannot be None')
        else:
            if not isinstance(X, collections.Iterable):
                raise ValueError('input must be an iterable\n')
            else:
                nx = 0
                distinct_values = set()
                Gs_ed, L = dict(), dict()
                for (i, x) in enumerate(iter(X)):
                    is_iter = isinstance(x, collections.Iterable)
                    if is_iter:
                        x = list(x)
                    if is_iter and len(x) in [0, 2, 3]:
                        if len(x) == 0:
                            warnings.warn('Ignoring empty element on index: '
                                          + str(i))
                            continue

                        elif len(x) in [2, 3]:
                            x = Graph(x[0], x[1], {}, self._graph_format)
                    elif type(x) is Graph:
                        x.desired_format("dictionary")
                    else:
                        raise ValueError('each element of X must have at ' +
                                         'least one and at most 3 elements\n')
                    Gs_ed[nx] = x.get_edge_dictionary()
                    L[nx] = x.get_labels(purpose="dictionary")

                    # Hold all the distinct values
                    distinct_values |= set(
                        v for v in itervalues(L[nx])
                        if v not in self._inv_labels[0])
                    nx += 1
                if nx == 0:
                    raise ValueError('parsed input is empty')

        nl = len(self._inv_labels[0])
        WL_labels_inverse = {dv: idx for (idx, dv) in
                             enumerate(sorted(list(distinct_values)), nl)}

        def generate_graphs(WL_labels_inverse, nl):
            # calculate the kernel matrix for the 0 iteration
            new_graphs = list()
            for j in range(nx):
                new_labels = dict()
                for (k, v) in iteritems(L[j]):
                    if v in self._inv_labels[0]:
                        new_labels[k] = self._inv_labels[0][v]
                    else:
                        new_labels[k] = WL_labels_inverse[v]
                L[j] = new_labels
                # produce the new graphs
                new_graphs.append([Gs_ed[j], new_labels])
            yield new_graphs

            for i in range(1, self._n_iter):
                new_graphs = list()
                L_temp, label_set = dict(), set()
                nl += len(self._inv_labels[i])
                for j in range(nx):
                    # Find unique labels and sort them for both graphs
                    # Keep for each node the temporary
                    L_temp[j] = dict()
                    for v in Gs_ed[j].keys():
                        credential = str(L[j][v]) + "," + \
                            str(sorted([L[j][n] for n in Gs_ed[j][v].keys()]))
                        L_temp[j][v] = credential
                        if credential not in self._inv_labels[i]:
                            label_set.add(credential)

                # Calculate the new label_set
                WL_labels_inverse = dict()
                if len(label_set) > 0:
                    for dv in sorted(list(label_set)):
                        idx = len(WL_labels_inverse) + nl
                        WL_labels_inverse[dv] = idx

                # Recalculate labels
                new_graphs = list()
                for j in range(nx):
                    new_labels = dict()
                    for (k, v) in iteritems(L_temp[j]):
                        if v in self._inv_labels[i]:
                            new_labels[k] = self._inv_labels[i][v]
                        else:
                            new_labels[k] = WL_labels_inverse[v]
                    L[j] = new_labels
                    # Create the new graphs with the new labels.
                    new_graphs.append([Gs_ed[j], new_labels])
                yield new_graphs

        if self._parallel is None:
            # Calculate the kernel matrix without parallelization
            self.km_test = np.sum((self.X[i].transform(g) for (i, g)
                       in enumerate(generate_graphs(WL_labels_inverse, nl))), axis=0)

        else:
            # Calculate the kernel marix with parallelization
            self.km_test = np.sum(self._parallel(joblib.delayed(etransform)(self.X[i], g) for (i, g)
                       in enumerate(generate_graphs(WL_labels_inverse, nl))), axis=0)

        mode = 'transform'
        K = self.non_linearity(kernel_function, mode)

        self._is_transformed = True
        if self.normalize:
            X_diag, Y_diag = self.diagonal()
            #print("Ydiag", Y_diag)
            if kernel_function is 'polynomial':
                Y_diag = np.power(Y_diag,2)
            #print("Xdiag", X_diag)
            #print("Xdiag", X_diag.shape)
            #print("Ydiag", Y_diag)
            #print("Ydiag", Y_diag.shape)
            old_settings = np.seterr(divide='ignore')
            #print("Divide km by sqrt(outer result) \n", np.divide(K, np.sqrt(np.outer(Y_diag, X_diag))))
            K = np.nan_to_num(np.divide(K, np.sqrt(np.outer(Y_diag, X_diag))))
            np.seterr(**old_settings)

        return K

    def diagonal(self):
        """Calculate the kernel matrix diagonal for fitted data.
        A funtion called on transform on a seperate dataset to apply
        normalization on the exterior.
        Parameters
        ----------
        None.
        Returns
        -------
        X_diag : np.array
            The diagonal of the kernel matrix, of the fitted data.
            This consists of kernel calculation for each element with itself.
        Y_diag : np.array
            The diagonal of the kernel matrix, of the transformed data.
            This consists of kernel calculation for each element with itself.
        """
        # Check if fit had been called
        check_is_fitted(self, ['X'])
        try:
            check_is_fitted(self, ['_X_diag'])
            if self._is_transformed:
                #print("selfX0", self.X[0])
                #print("selfX0", self.X[0].diagonal())
                ##print("selfX0", self.X[0].diagonal()[1])
                #print("selfX0 shape", self.X[0].diagonal()[1].shape)
                Y_diag = self.X[0].diagonal()[1]
                for i in range(1, self._n_iter):
                    Y_diag += self.X[i].diagonal()[1]
        except NotFittedError:
            print("EXCEPTION: NOT FITTED")
            # Calculate diagonal of X
            if self._is_transformed:
                X_diag, Y_diag = self.X[0].diagonal()
                # X_diag is considered a mutable and should not affect the kernel matrix itself.
                X_diag.flags.writeable = True
                for i in range(1, self._n_iter):
                    x, y = self.X[i].diagonal()
                    X_diag += x
                    Y_diag += y
                self._X_diag = X_diag
            else:
                # case sub kernel is only fitted
                X_diag = self.X[0].diagonal()
                # X_diag is considered a mutable and should not affect the kernel matrix itself.
                X_diag.flags.writeable = True
                for i in range(1, self._n_iter):
                    x = self.X[i].diagonal()
                    X_diag += x
                self._X_diag = X_diag

        if self._is_transformed:
            return self._X_diag, Y_diag
        else:
            return self._X_diag


def efit(object, data):
    """Fit an object on data."""
    object.fit(data)


def efit_transform(object, data):
    """Fit-Transform an object on data."""
    return object.fit_transform(data)


def etransform(object, data):
    """Transform an object on data."""
    return object.transform(data)
