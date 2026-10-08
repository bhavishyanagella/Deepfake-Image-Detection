
# CNN-LSTM Method

A convolutional neural network (CNN) extracts spatial representations from images.
ResNet50 is a residual CNN that can be initialized with ImageNet weights and
fine-tuned for a target classification task.

An LSTM is designed for sequential data. In video deepfake detection, CNN features
from consecutive frames can be provided to an LSTM so it can learn temporal
relationships.

For an image-only CNN-LSTM experiment, an image can be divided into multiple
spatial regions. The CNN extracts one feature vector for each region, and the LSTM
processes the ordered region sequence. This is more meaningful than reshaping a
single feature vector into a one-timestep sequence.
