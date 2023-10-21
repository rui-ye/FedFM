# FedFM  algorithm on mindspore

## Introduction

This code transfers FedFM algorithm from torch version to mindspore version and verifies that it also works well on mindspore. To be more specific, FedFM has an increase about 5%(from 65% to 70%) over traditional FedAVG algorithm in validation and testing accuracy.

## Environment

We use mindspore in the environment as follows.

```python
asttokens           2.2.1
astunparse          1.6.3
Brotli              1.0.9
certifi             2023.7.22
charset-normalizer  3.2.0
contourpy           1.1.0
cycler              0.11.0
decorator           5.1.1
download            0.3.5
fonttools           4.41.1
idna                3.4
importlib-resources 6.0.0
joblib              1.3.1
kiwisolver          1.4.4
matplotlib          3.7.2
mindspore           2.0.0
numpy               1.25.1
packaging           23.1
Pillow              10.0.0
pip                 23.2.1
platformdirs        3.9.1
pooch               1.7.0
protobuf            4.23.3
psutil              5.9.5
pyparsing           3.0.9
PySocks             1.7.1
python-dateutil     2.8.2
PyYAML              6.0.1
requests            2.31.0
scikit-learn        1.3.0
scipy               1.11.1
setuptools          68.0.0
six                 1.16.0
threadpoolctl       3.2.0
tqdm                4.65.0
typing_extensions   4.7.1
urllib3             2.0.4
wheel               0.41.0
zipp                3.16.2
```

We strongly recommend using mindspore version 2.0 instead of other versions, which may trigger unknown error.

To run mindspore successfully, you may need to add command

```python
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:{YOUR_ENVIRONMENT_PATH}/lib
```

before running mindspore.

## Run this code

To run this code, you can directly use

```python
sh fedfm_10_1.sh
```

You can also change parameters in the bash file. We recommend batch size over 64, which makes accuracy better.