# Tree-KGQA
PyTorch code for the IEEE Access paper: Tree-KGQA: An Unsupervised Approach for Question Answering Over Knowledge Graphs [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9770789).


[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## ⚙️ Installation
Required : [Anaconda](https://www.anaconda.com/products/individual)

```commandline
conda create -n treekgqa -y python=3.8 && source activate treekgqa
pip install -r requirements.txt

chmod +x setup.sh
./setup.sh
```
## 🔧 Pre-processing

We use Wikidata entities provided by [ELQ](https://arxiv.org/pdf/2010.02413.pdf) . In order to perform inference, first index the Wikidata entities by executing the follwing command:
```python
python utils/indexer.py --output_path data/wikidata/indexed_wikidata_entities.pkl --faiss_index hnsw --save_index
```
The indexing might take few hours depending on your system capabilities and resource.

## 🎯 Inference
To test entity linking on the ```webqsp``` dataset run the following command:

```python
python -u run_kgqa.py --dataset webqsp --task EL --use_api --use_indexing --QAtype complex --evaluate
```

## 📜 License
[MIT](https://github.com/rashad101/Tree-KGQA/blob/main/LICENSE.md)

## 📝 Citation
```text
@ARTICLE{9770789,
    author={Rony, Md Rashad Al Hasan and Chaudhuri, Debanjan and Usbeck, Ricardo and Lehmann, Jens},
    journal={IEEE Access}, 
    title={Tree-KGQA: An Unsupervised Approach for Question Answering Over Knowledge Graphs}, 
    year={2022},
    volume={10},
    number={},
    pages={50467-50478},
    doi={10.1109/ACCESS.2022.3173355}
  }
```
