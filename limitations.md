
# Limitations and Responsible Use

Deepfake detection is an evolving problem. A detector trained on one dataset may
not generalize to new datasets or future synthesis methods.

Predictions should be presented as model probabilities or scores, not definitive
proof. Human review and provenance evidence may be required for high-stakes use.

RAG does not automatically improve the classifier's accuracy. Its main role in
this project is to retrieve documented knowledge and improve explanation.

Classifier evaluation must therefore be measured independently from the RAG
component.
