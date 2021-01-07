[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
[![PyPI version](https://img.shields.io/pypi/v/mudes?color=%236ecfbd&label=pypi%20package&style=flat-square)](https://pypi.org/project/mudes/)
[![Downloads](https://pepy.tech/badge/mudes)](https://pepy.tech/project/mudes)
# MUDES - {Mu}ltilingual {De}tection of Offensive {S}pans

We provide state-of-the-art models to detect toxic spans in text. We have evaluated our models on  Toxic Spans task at SemEval 2021 (Task 5).

## Installation
You first need to install PyTorch. The recommended PyTorch version is 1.6.
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, you can install MUDES from pip. 

#### From pip

```bash
pip install mudes
```

## Pretrained MUDES Models

We will be keep releasing new models. Please keep in touch. We have evaluated the models on the trial set released for Toxic Spanstask at SemEval 2021.

| Models               | Average F1    |
|----------------------|:-------------:|
| en-base              | 0.6734        |
| en-large             | 0.6886        |
| multilingual-base    | 0.5953        |
| multilingual-large   | 0.6013        |

## Prediction
Following code can be used to predict toxic spans in text. Upon executing, it will download the relevant model and return the toxic spans.   

```python
from mudes.app.mudes_app import MUDESApp

app = MUDESApp("en-large", use_cuda=False)
print(app.predict_toxic_spans("You motherfucking cunt", spans=True))
```


