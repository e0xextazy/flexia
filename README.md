<img src="https://i.ibb.co/Qrvyg8S/Gallery-1649448825801.png" alt="Gallery-1649448825801" border="0">

Torch Components is an open-source high-level API above the PyTorch framework. It provides functionality, which can be easily inserted into any PyTorch training and validation scripts.

# Installation

```
git clone -q https://github.com/vad13irt/torch-components.git
```

```py
import sys
sys.path.append('./torch-components/')
import torch_components
```


# API Reference

### Averager
Averager calculates some statistics (sum, average, and count) for given values.

### Timer
Timer calculates how much time was elapsed after initializing an instance of the class, and also calculates how much time remains for a certain fraction of the process.

### Configuration
Configuration is an analog of TrainingArguments or Arguments from fast.ai and HuggingFace Transformers respectively, it provides a good interface for saving configuration variables.


## Callbacks
### Early Stopping
Early Stopping is a simple method of regularization used to avoid overfitting during training Machine Learning models by stopping the training when the monitored value has not been improved.

### Model Checkpoint
Model Checkpoint saves the model's weights (optimizer's state, and scheduler's state) when the monitored value has been improved.

### Timing
Timing stops training when the duration of the training stage reaches a certain limit of time. It is useful when you are using time-limit sources, e.g. Google Colab or Kaggle Kernels/Notebooks.


# Contribution
Torch Components is open for contributions and issues! We are waiting for your first contribution! 

