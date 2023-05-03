# CSCI-585-Image-to-sentence-generation

![alt text](https://github.com/Batyr1203/CSCI-585-Image-to-sentence-generation/blob/main/images/demo.png?raw=true)

Spring23, Computer Vision Course term project on Image Captioning

Term project presents 2 models that are composed of a Convolutional Network at the encoding step, and 2 variants of a language model: Recurrent Network one and Transformer-based one. We utilized three common datasets for this task: MSCOCO, Flickr8K, FLickr30K, and evaluated the performance with BLEU metrics.

### Note
The implementation is based on `Python3.10` the new `Pytorch 2.0` release.


### Requirements
```h5py==3.8.0
imageio==2.28.1
matplotlib==3.7.1
nltk==3.8.1
opencv-python==4.7.0.72
scikit-image==0.20.0
streamlit==1.22.0
torch==2.0.0
torchvision==0.15.1
tqdm==4.65.0
```

### Inference
Run demo app for the project: \
`pip install -r requirements.txt` \
`streamlit run demo_app.py`


### Performance of the Baseline (LSTM) and the Main model (Transformer)
(Beam size = 5)

Flickr8k
| Metric          | LSTM   | Transformer |
|-----------------|--------|-------------|
| BLEU-1          | 0.6517 | 0.6445      |
| BLEU-2          | 0.4715 | 0.4654      |
| BLEU-3          | 0.3357 | 0.3290      |
| BLEU-4          | 0.2310 | 0.2284      |

Flickr30k
| Metric          | LSTM   | Transformer |
|-----------------|--------|-------------|
| BLEU-1          | 0.6949 | 0.6838      |
| BLEU-2          | 0.5137 | 0.5004      |
| BLEU-3          | 0.3780 | 0.3611      |
| BLEU-4          | 0.2696 | 0.2587      |

MSCOCO
| Metric          | LSTM   | Transformer |
|-----------------|--------|-------------|
| BLEU-1          | 0.7562 | 0.7420      |
| BLEU-2          | 0.5936 | 0.5813      |
| BLEU-3          | 0.4604 | 0.4482      |
| BLEU-4          | 0.3519 | 0.3464      |


## Acknowlegments
- The Baseline of the code is taken from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

- The Transformer-based model is adapted from https://github.com/RoyalSkye/Image-Caption
