# NLP Text Similarity Tools

A small Python project for constructing sparse word co-occurrence matrices,
training a GloVe-style embedding model, and querying cosine similarity.
A self-contained example and automated tests make the core pipeline runnable
without existing model files, private data, or automatic corpus downloads.

## Quick start

Use Python 3.12 or 3.13 in a virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python train_model.py --seed 42 --output artifacts
python -m pytest -q
```

The included `examples/corpus.txt` contains synthetic sentences about research
and software. The command builds a fresh dictionary, trains the model, queries
neighbors for `research`, and writes `corpus.pkl`, `model.pkl`, and `summary.json`.
These generated artifacts are ignored by Git. Repeating the command with the
same inputs, seed, arguments, and numerical environment produces the same model.
Results are not promised to be bitwise identical across different numerical
libraries or hardware.

```sh
python train_model.py --corpus examples/corpus.txt --window 3 --dim 16 --epochs 50 --query software --output artifacts
```

## Model and engineering decisions

The original project uses **one embedding and one bias per word**, shared across
both endpoints of an unordered co-occurrence pair. This is a small GloVe-style
variant, not an exact implementation of Stanford GloVe's separate word/context
parameter tables. The weighted least-squares idea comes from
[Pennington, Socher, and Manning's GloVe work](https://nlp.stanford.edu/projects/glove/).

For each observed pair, the objective is
`0.5 * f(count) * (dot(w_i, w_j) + b_i + b_j - log(count))^2`, where
`f(count) = min(1, count / max_count)^alpha`.
Both vector gradients use the same pre-update values. AdaGrad accumulates
squared gradients separately, and both word biases are updated. `max_loss`
retains the original parameter name but bounds the weighted residual used by
the optimizer, not the reported objective. No claim of monotonic improvement
on arbitrary datasets is made.

Co-occurrences use an inverse-distance context window within each input line.
Unordered pairs are stored in the upper triangle of a SciPy COO matrix, with
self-word pairs omitted. Tokenization uses NLTK's regular-expression-based
Treebank tokenizer. Seed zero is valid, each `Corpus` owns its own dictionary,
and model queries explicitly exclude the queried word even when scores tie.

## What was repaired

The training loop previously updated the first word's bias twice, calculated
the second vector's gradient after changing the first vector, and did not use
the supplied seed. Training also depended on absent input/model files, and
imports triggered downloads and example execution. The maintenance changes fix
those paths, validate core inputs, and add a clean-checkout training CLI.

## Verification

```sh
python -m pytest -q
python -m compileall -q -x '/\.git/' .
```

Tests include a hand-computed optimizer step, finite-difference gradients,
repeatability with seed zero, input validation, exact co-occurrence fixtures,
query tie handling, pickle round trips, side-effect-free imports, and a CLI run
from a temporary directory. CI checks Python 3.12 on Linux and macOS.

The tiny corpus is a **correctness demonstration**, not a semantic accuracy or
performance benchmark. A lower training objective does not establish useful
real-world word representations.

## Reusing a trained model

```python
from GloVe import GloVe
model = GloVe.load_file("artifacts/model.pkl")
print(model.get_most_similar("research", number=5))
```

**Only load pickle files that you trust.** Pickle can execute code while loading;
these helpers are not safe deserializers for files received from strangers.

## Other retained utilities

`color_docx.py` and `extend_docx.py` are the original document-highlighting
helpers, separate from the tested training path. They are not a general-purpose
paper-generation system. `try.py` is the retained, optional Doc2Vec experiment;
its Python 2 code was replaced with a small Python 3 example using the included
synthetic corpus. Its modeling results are not covered by the GloVe tests. It
additionally needs `gensim`, which is not required by the quick start.

`sp_remover(tokens, stop_words=...)` accepts a supplied stop list for offline
use. Omitting it uses the NLTK English stopword corpus, which must be installed
explicitly with `python -m nltk.downloader stopwords`. Imports never download it.

## Limitations

The trainer is intended for small educational corpora. It has no large-scale
training, pre-trained embeddings, validated biomedical results, or semantic
accuracy claims. The original module names are retained for compatibility.
