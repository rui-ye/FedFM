# FedFM  algorithm on Pytorch

## Introduction

This code is the torch version of FedFM algorithm. FedFM has an increase about 6%(from 66% to 72%) over traditional FedAVG algorithm in validation and testing accuracy on CIFAR10 dataset.

## Environment

We use torch in the environment as follows.

```python
catalogue           2.0.9              
certifi             2016.9.26          
charset-normalizer  2.0.12             
click               8.0.4              
cycler              0.11.0             
dataclasses         0.8                
idna                3.4                
importlib-metadata  4.8.3              
importlib-resources 5.4.0              
joblib              1.1.1              
kiwisolver          1.3.1              
matplotlib          3.3.4              
murmurhash          1.0.9              
numpy               1.19.5             
pandas              1.1.5              
Pillow              8.4.0              
pip                 20.0.2             
pyparsing           3.0.7              
python-dateutil     2.8.2              
pytz                2023.3.post1       
scikit-learn        0.24.2             
scipy               1.5.4              
seaborn             0.11.2             
setuptools          49.6.0.post20210108
six                 1.16.0             
spacy-legacy        3.0.12             
srsly               2.4.7              
threadpoolctl       3.1.0              
torch               1.8.1              
torchvision         0.9.1+cu111        
tqdm                4.64.1             
typer               0.9.0              
typing-extensions   3.10.0.2           
ujson               4.3.0              
urllib3             1.26.16            
wheel               0.36.2             
zipp                3.6.0   
```

## Run this code

To run this code, you can directly use

```python
sh fedfm_10_1.sh
```

To set the communication round to start using feature matching, you can change the variable

```python
start_ep_fm
```

in the shell file start_ep_fm, we also set a default value for this variable.

You can also change parameters in the shell file.