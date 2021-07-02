from setuptools import setup
from setuptools import find_packages
import subprocess

# conda create --name env_yield_prediction python=3.6.12 -y

# python setup.py install

# conda-forge
bashCommand1 = "conda config --add channels conda-forge"
process = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

bashCommand2 = "conda config --set channel_priority strict"
process = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Instalation requirements
setup(name='yield_prediction',
      version='1.0',
      install_requires=['grakel==0.1.8',
                        'matplotlib==3.3.2',
                        'xlrd==1.2.0',
                        'openpyxl==3.0.5',
                        'scikit-learn==0.22.1',
                        'numpy==1.19.1',
                        'scipy==1.5.2',
                        'pandas==1.1.1'
                        ],
      package_data={'yield_prediction': ['README.md']},
      packages=find_packages())

bashCommand3 = "conda install -n env_yield_prediction rdkit=2021.03.3 -y"
process = subprocess.Popen(bashCommand3.split(), stdout=subprocess.PIPE)
output, error = process.communicate()


