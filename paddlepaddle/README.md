# FedFM  algorithm on paddlepaddle

## Introduction

This code transfers FedFM algorithm from torch version to paddle version and verifies that it also works well on paddle. To be more specific, in the first scene we defined, FedFM has an increase about 5% over traditional FedAVG algorithm in validation accuracy, as well as an increase about 3% over FedAVG in test accuracy. In the second scene we defined, FedFM has an increase about 3.5% over FedAVG in both validation and test accuracy. 

## Environment

We use paddle in the environment as follows.

```python
astor              0.8.1
certifi            2023.5.7
charset-normalizer 3.1.0
cycler             0.11.0
decorator          5.1.1
fonttools          4.38.0
idna               3.4
joblib             1.3.1
kiwisolver         1.4.4
matplotlib         3.5.3
numpy              1.21.6
opt-einsum         3.3.0
packaging          23.1
paddle-bfloat      0.1.7
paddlepaddle-gpu   2.4.2.post112
Pillow             9.2.0
pip                23.1.2
protobuf           3.20.0
pyparsing          3.1.0
PySocks            1.7.1
python-dateutil    2.8.2
requests           2.31.0
scikit-learn       1.0.2
scipy              1.7.3
setuptools         68.0.0
six                1.16.0
threadpoolctl      3.1.0
typing_extensions  4.7.1
urllib3            2.0.3
wheel              0.40.0
```

To run paddlepaddle-gpu successfully, you may need to add command

```python
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:{YOUR_ENVIRONMENT_PATH}/lib
```

before running paddle.

## Run this code

To run this code, you can directly use

```python
sh run_scene1.sh
sh run_scene2.sh
```

To set the communication round to start using feature matching, you can change the variable

```python
start_ep_fm
```

in the shell file start_ep_fm, we also set a default value for this variable.

You can also change parameters in the shell file.