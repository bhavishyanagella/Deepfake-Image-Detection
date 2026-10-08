
# Evaluation

Recommended metrics include accuracy, precision, recall, F1-score, ROC-AUC, and
a confusion matrix.

Use an independent test set when claiming generalization. A detector can learn
dataset-specific shortcuts.

Report class counts, train/validation/test methodology, random seed, image size,
optimizer, learning rate, batch size, epochs, early stopping, and hardware.

A high confidence score is not equivalent to certainty. The detector may fail on
unseen manipulation methods, new generators, heavy compression, resizing, crops,
or distribution shifts.
