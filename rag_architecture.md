
# RAG Architecture

Retrieval-Augmented Generation (RAG) separates knowledge retrieval from the
classifier.

The classifier first produces a prediction and probabilities. A retriever searches
a curated knowledge base for relevant passages. A language model can then generate
a grounded explanation using those retrieved passages.

The RAG layer should not override the classifier. It should explain the result,
provide documented background information, and communicate limitations.

A typical pipeline is:

classifier -> query construction -> embedding model -> vector database ->
top-k evidence -> language model -> explanation.

FAISS can be used for local vector similarity search. Sentence-transformers can
generate dense embeddings. The knowledge base should contain trusted research
papers, project documentation, evaluation notes, and model limitations.
