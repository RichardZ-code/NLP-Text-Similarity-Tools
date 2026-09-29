# NLP Text Similarity Tools

Small Python tools for word co-occurrence modeling, cosine-similarity queries,
and highlighted Word-document reports. The core is a **tied-embedding GloVe-style
model**, not a wrapper around an LLM and not Stanford's full GloVe implementation.

## Quick start

Use Python 3.12 or 3.13. The pinned direct dependencies are in `requirements.txt`.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python train_model.py --seed 0 --query research
```

This reads the included `examples/corpus.txt`, builds a sparse co-occurrence matrix,
trains from scratch, and writes `artifacts/corpus.pkl` and `artifacts/model.pkl`.
No pre-existing model, private corpus, downloaded word vectors, API key, or NLTK
corpus download is required. The tiny example demonstrates the pipeline, not
validated semantic accuracy. `python try.py` runs the same example.

To use your own UTF-8 corpus, put one context per line:

```sh
python train_model.py --corpus path/to/text.txt --dim 32 --epochs 50 --window 5 --seed 0 --output artifacts/custom
```

## How it works

1. `Corpus.py` tokenizes each context and constructs inverse-distance-weighted
   sparse counts for distinct word pairs within the selected window.
2. `GloVe.py` fits a weighted log-co-occurrence objective using AdaGrad-style updates.
3. A saved dictionary maps words to embedding rows for cosine-similarity queries.
4. `color_docx.py` and `extend_docx.py` provide standalone highlighting helpers.

The model preserves this project's original shared word/context vector table.
Each unordered off-diagonal pair is stored once. The objective per pair is
`0.5 * weight * (dot(w_i, w_j) + b_i + b_j - log(count))**2`, with clipped weighted
residuals during updates. Both vector gradients use pre-update values, and both
biases are updated separately. This differs from canonical GloVe's separate word
and context tables. See the [Stanford GloVe project](https://nlp.stanford.edu/projects/glove/)
for the original method and implementation.

## Query a trained model

```python
from GloVe import GloVe
model = GloVe.load_file("artifacts/model.pkl")
print(model.get_most_similar("research", number=5))
print(model.check_similarity("research", "data"))
```

**Pickle files must come from a trusted source.** Never load a stranger's model or
corpus pickle. The persistence format is retained for compatibility, not safe
interchange with untrusted users.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover a hand-calculated one-step gradient, both bias updates, seed-zero
reproducibility, toy-objective reduction, malformed inputs, independent corpus
instances, exact tiny co-occurrence counts, missing/tied/zero-vector queries,
trusted-file round trips, offline imports, document edge cases, and CLI execution.
CI also compiles active Python modules and runs the bundled example.

## Maintenance changes

- Fixed duplicate updates to one word's bias and stale-vector gradient ordering.
- Made `seed=0` effective and repeatable within a fixed runtime/dependency setup.
- Removed shared mutable corpus defaults and import-time downloads/example runs.
- Replaced the missing-input training script with a runnable CLI and tiny corpus.
- Added explicit self-exclusion in nearest-neighbor results and zero-vector handling.
- Added document input checks without changing existing highlight thresholds.
- Preserved the old Python 2 Doc2Vec sketch in `legacy/doc2vec_python2.py.txt`.
  It is reference material, not a supported executable or part of the GloVe pipeline.

## Limits

This is educational word-level similarity software, not an evaluated sentence
similarity system, plagiarism detector, or academic-paper generator. Training is
single-process and intended for small corpora. Identical seeds do not promise
bitwise identical results across all dependency versions or hardware. Optional
`sp_remover` calls need NLTK's stopwords data unless a `stop_words` collection is
passed explicitly; installation is never triggered on import. Existing report
helpers use fixed output names by default and can overwrite those local files.
