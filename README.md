# yield_prediction

Machine Learning applied in Chemistry to solve the reaction yield prediction problem. First task: application of non-linearity in the Weisfeiler-Lehman graph kernel in order to improve the measure of comparison between molecules and thus enhance the complexity of the support vector regression models. Second task: creation of a deep learning base to solve this problem through graph neural networks, with the extraction of molecular representations using graph convolutional layers and global read-out operations.

Extension of the project developed by Haywood et al.<sup>1</sup> (__[yield_prediction](https://github.com/alexehaywood/yield_prediction)__: Prediction of reaction yields using support vector regression models built on structure-based and quantum chemical descriptors).

## Dependencies
* Python (3.6)
* GraKel-dev (0.1.8)
* RDKit (2021.03.3)
* Matplotlib (3.3.2)
* xlrd (1.2.0)
* openpyxl (3.0.5)
* Scikit Learn (0.22.1)
* Numpy (1.19.1)
* Scipy (1.5.2)
* Pandas (1.1.1)
* PyTorch (1.9.0 + CUDA 11.1 (Linux) / 1.9.0 + CPU (OS X))
* PyTorch Geometric (1.7.2) & dependencies
* XlsxWriter (3.0.1)

### Environment creation
```bash
conda create --name yield_prediction python=3.6 -y

python setup.py install
```

# Instructions

### Preprocessing

The data and quantum chemical descriptors in `yield_prediction/data/original` are from the open-source dataset published by Doyle et al.<sup>2</sup> (__[rxnpredict](https://github.com/doylelab/rxnpredict)__).

The Doyle et al.<sup>2</sup> dataset is preprocessed using `yield_prediction/assemble_rxns.py`. The molecules in each reaction and corresponding yield data can be found in `yield_prediction/original/reactions`.

The SVR preprocessing is from Haywood et al.<sup>1</sup>. Some changes were included for the GNN preprocessing due to its different algorithmic nature.

### Model Development
Models are trained and tested using `run_ml_out-of-sample.py`.

### Predictions
The SVR predictions for the out-of-sample tests were collated using `gather_results.py`. The collection of predictions by the GNN models is incorporated in the development. For a broader analysis of deep learning, we developed `gather_gnn_results.ipynb`. All results can be found in `yield_prediction/results`.


## References
[1] A. Haywood, J. Redshaw, M. Hanson-Heine, A. Taylor, A. Brown, A. Mason, T. Gaertner, and J. Hirst, 2021.

[2] D. T. Ahneman, J. G. Estrada, S. Lin, S. D. Dreher and A. G. Doyle, *Science*, 2018, **360**, 186–190.
