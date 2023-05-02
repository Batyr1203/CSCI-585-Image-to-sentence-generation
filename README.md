# CSCI-585-Image-to-sentence-generation
Spring23, Computer Vision Course term project on Image Captioning

This repository consists of models presented as a term project.

Term project presents 2 models that are composed of a Convolutional Network at the encoding step, and different variants of language model: Recurrent Network one and Transformer based one. We utilized three common datasets for this task: MSCOCO, Flickr8K, FLickr30K, and evaluated the performance with BLEU metrics.

### Note
The implementation is based on the new `Pytorch 2.0` release.

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


## Acknolegments
The Baseline of the code is taken from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

The Transformer-based model is adapted from https://github.com/RoyalSkye/Image-Caption
