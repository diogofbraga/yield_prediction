from setuptools import setup
from setuptools import find_packages
import subprocess
from sys import platform

# conda-forge
bashCommand1 = "conda config --add channels conda-forge pytorch ostrokach-forge rusty1s"
process = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

bashCommand2 = "conda config --set channel_priority strict"
process = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Instalation requirements
setup(name='yield_prediction',
      version='1.0',
      install_requires=['networkx==2.2',
                        'grakel==0.1.8',
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

bashCommand3 = "conda install -n yield_prediction rdkit=2021.03.3 -y"
process = subprocess.Popen(bashCommand3.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

if platform == "linux" or platform == "linux2": # linux
      bashCommand4 = "pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html"
      process = subprocess.Popen(bashCommand4.split(), stdout=subprocess.PIPE)
      output, error = process.communicate()

      bashCommand5 = "pip3 install torch-scatter==2.0.8 torch-sparse==0.6.11 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==1.7.2 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html"
      process = subprocess.Popen(bashCommand5.split(), stdout=subprocess.PIPE)
      output, error = process.communicate()
elif platform == "darwin": # OS X
      bashCommand4 = "pip3 install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0"
      process = subprocess.Popen(bashCommand4.split(), stdout=subprocess.PIPE)
      output, error = process.communicate()

      bashCommand5 = "pip3 install torch-scatter==2.0.8 torch-sparse==0.6.11 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==1.7.2 -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html"
      process = subprocess.Popen(bashCommand5.split(), stdout=subprocess.PIPE)
      output, error = process.communicate()


