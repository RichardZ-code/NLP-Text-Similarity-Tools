# NLP Text Similarity Tools

Small Python tools for sparse word co-occurrence construction, GloVe-style embedding
training, cosine-similarity queries, and Word-document highlighting.

## Quick start

Use Python 3.12 or 3.13. From a checkout of this repository:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python train_model.py
python -m pytest -q
```

The training command uses the included `examples/corpus.txt`, starts with an empty
vocabulary, and writes `artifacts/corpus.pkl`, `model.pkl`, and `report.json`.
It does not require `traindata.txt`, a previously trained model, an NLTK data
download, an API key, or a paid service. Installation itself requires package access.
The sample is synthetic developer-workflow prose, not clinical or employer data.

```sh
python train_model.py --corpus examples/corpus.txt --seed 0 --epochs 100 --query data
```

The report includes vocabulary size, sparse-pair count, initial/final training loss,
and nearest neighbors. It is a reproducible toy demonstration, **not an embedding
accuracy benchmark**. Same-environment runs with a fixed seed are reproducible;
bit-identical output across different platforms/library versions is not promised.

## Model and scope

This keeps the original project's **single, shared embedding table** and unordered,
off-diagonal co-occurrence pairs. The objective for a pair is
`0.5 * weight * (dot(w_i, w_j) + b_i + b_j - log(count)) ** 2`.
AdaGrad-style updates use separate accumulated squared gradients, with a clipped
weighted-residual coefficient controlled by the legacy `max_loss` parameter.
Both vector derivatives use pre-update vectors, and both word biases are updated.

This is a GloVe-inspired educational variant, not a reimplementation of Stanford's
canonical separate word/context tables. See the [original GloVe project](https://nlp.stanford.edu/projects/glove/).
The corpus representation stores each unordered pair once, applies inverse-distance
window weights, and excludes same-word pairs. Large-corpus optimization and semantic
quality evaluation are outside this maintenance patch.

## Files

- `Corpus.py`: vocabulary and sparse co-occurrence construction.
- `GloVe.py`: deterministic training, loss calculation, and similarity queries.
- `lower_tokenizer.py`: offline Treebank tokenization and stemming.
- `train_model.py`: runnable, configurable training example.
- `color_docx.py`, `extend_docx.py`: original document-highlighting utilities.
- `try.py`: compatibility entry point for the same tested training demo. The old
  Python 2 Doc2Vec scratch example is preserved in Git history, not presented as
  a supported alternative model. It no longer requires Gensim or scikit-learn.

Stopword removal accepts an explicit `stop_words` iterable. Using NLTK's built-in
English list instead requires an explicit `python -m nltk.downloader stopwords`.
Importing the core modules never downloads corpora or starts training.

## Verification

Tests check an independent finite-difference gradient, both bias updates, simultaneous
vector gradients, seed zero, independent vocabularies, hand-calculated sparse counts,
invalid inputs, decreasing fixture loss, identity-filtered similarity, trusted-file
round trips, and complete CLI runs from another working directory. CI runs these on
Python 3.12 and 3.13 and checks every checked-in Python file for valid syntax.
The original document-formatting utilities are retained, not comprehensively audited.

## Saved files and provenance

Legacy `save_file` / `load_file` use Python pickle. **Load only files you created or
otherwise fully trust**: unpickling an attacker-controlled model can execute code.
Generated models and reports are ignored by Git. Preserve existing collaborator
credits and confirm rights before adding any external research corpus or employer data.
