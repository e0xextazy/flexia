![Flexia logo](images/flexia_logo.png)

Flexia (Flexible Artificial Intelligence) is an open-source library, which provides high-level functionality for developing accurate Deep Learning models. There is a variety of methods for controlling (e.g Early Stopping, Model Checkpointing, Timing, etc.) and monitoring (e.g Weights & Biases, Print, Logging, and Taqadum) training, validation, and inferencing loops respectively.

API interface is designed as HuggingFace Transformers, and PyTorch Lightning frameworks have, so the users, who already used one of them can faster adapt to Flexia's API.


## Installation

Flexia offers you many ways to do it depending on your setup: PyPi (pip), Google Colab, and Kaggle Kernels.

#### PyPi (pip)

```py
pip install flexia 
```

#### Google Colab

```
!git clone https://github.com/vad13irt/flexia.git
```

```py
import sys
sys.path.append("./flexia")
import flexia
```

#### Kaggle Kernels

There are many ways to install libraries in Kaggle Kernels. We will describe only one way, which is faster and more suitable for  Kernels Competitions, where submitting require disabling an Internet connection.


## Getting Started

## Examples

In order to speed up learning Flexia's API, we offer you to look at some examples. The examples cover most Deep Learning tasks such as Classification, Regression, Semantic Segmentation, Object Detection, and Learning to Rank.

- [Digit Recognizer](examples/Digit%20Recognizer/)
- [Carvana Image Segmentation](examples/Carvana%20Image%20Masking%20Challenge/)
- [Global Wheat Detection](examples/Global%20Wheat%20Detection/)

## Contribution

Flexia is always open to your Pull-Requests (PRs) and issues! Pull-Requests are required to have clean and readable code, thorough testing, and well-described documentation.

## Releases

## TO-DO

- [ ] Documentation (readthedocs)
- [ ] PyPi
- [ ] Distributed Training
- [ ] DeepSpeed integration
- [ ] Callbacks
    - [ ] Model Checkpoint
    - [ ] Early Stopping
    - [ ] Timing
    - [x] Lambda
- [ ] Inferencer
- [ ] Torch XLA
- [ ] Examples
    - [ ] Digit Recognizer
    - [ ] Global Wheat Detection
    - [ ] Carvana Image Masking Challenge
- [ ] Tests
