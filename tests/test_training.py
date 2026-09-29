import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from Corpus import Corpus, prepare_data
from GloVe import GloVe, cosine
from lower_tokenizer import lower_tokenizer, sp_remover, word_stemmer

ROOT = Path(__file__).resolve().parents[1]


def pair(count=8.0):
    return sp.coo_matrix(([count], ([0], [1])), shape=(2, 2))


def test_one_step_uses_both_old_vectors_and_updates_both_biases():
    seed, lr = 7, 0.02
    initial = (np.random.RandomState(seed).rand(2, 3) - 0.5) / 3
    derivative = (8.0 / 100) ** 0.5 * (np.dot(initial[0], initial[1]) - np.log(8.0))
    gradients = np.array([derivative * initial[1], derivative * initial[0]])
    model = GloVe(dim=3, seed=seed, learning_rate=lr).fit_vectors(pair(), epochs=1)
    np.testing.assert_allclose(model.word_vectors, initial - lr * gradients, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(model.word_biases, [-lr * derivative] * 2, rtol=1e-12)
    np.testing.assert_allclose(model.vectors_sum_gradients, 1 + gradients**2)
    np.testing.assert_allclose(model.biases_sum_gradients, [1 + derivative**2] * 2)


def test_gradients_match_finite_differences():
    model = GloVe(dim=2, seed=3).fit_vectors(pair(), epochs=0)
    before = model.word_vectors.copy()
    gradient = np.zeros_like(before)
    step = 1e-6
    for index in np.ndindex(before.shape):
        model.word_vectors[index] = before[index] + step
        plus = model.objective(pair())
        model.word_vectors[index] = before[index] - step
        minus = model.objective(pair())
        model.word_vectors[index] = before[index]
        gradient[index] = (plus - minus) / (2 * step)
    model._pair_step(0, 1, 8.0)
    np.testing.assert_allclose((before - model.word_vectors) / model.learning_rate, gradient, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 42])
def test_seed_reproducibility(seed):
    a = GloVe(seed=seed).fit_vectors(pair(), epochs=10)
    b = GloVe(seed=seed).fit_vectors(pair(), epochs=10)
    np.testing.assert_array_equal(a.word_vectors, b.word_vectors)
    np.testing.assert_array_equal(a.word_biases, b.word_biases)


def test_toy_objective_decreases():
    model = GloVe(seed=42).fit_vectors(pair(), epochs=30)
    assert model.loss_history[-1] < model.loss_history[0]
    assert all(np.isfinite(model.loss_history))


@pytest.mark.parametrize("matrix", [
    sp.coo_matrix((2, 2)), pair(-1.0), pair(float("nan")), pair(float("inf")),
    sp.eye(2, format="coo"), sp.coo_matrix((2, 3)), np.eye(2),
])
def test_rejects_invalid_training_input(matrix):
    with pytest.raises(ValueError):
        GloVe(seed=0).fit_vectors(matrix)


@pytest.mark.parametrize("kwargs", [{"dim": 0}, {"learning_rate": 0}, {"seed": -1}, {"alpha": 2}, {"max_loss": float("nan")}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        GloVe(**kwargs)


def test_independent_corpus_dictionaries():
    a, b = Corpus(), Corpus()
    a.fit_matrix([["first", "second"]])
    assert b.dictionary == {}
    existing = {"first": 0}
    c = Corpus(existing)
    c.fit_matrix([["first", "second"]])
    assert existing == {"first": 0}


def test_cooccurrence_values_and_boundaries():
    corpus = Corpus()
    corpus.fit_matrix([["a", "b", "c"], ["c", "a"]], window_size=2)
    assert corpus.dictionary == {"a": 0, "b": 1, "c": 2}
    np.testing.assert_array_equal(corpus.matrix.toarray(), [[0, 1, 1.5], [0, 0, 1], [0, 0, 0]])


def test_no_self_pairs():
    corpus = Corpus()
    corpus.fit_matrix([["a", "a", "b"]], window_size=2)
    np.testing.assert_array_equal(corpus.matrix.toarray(), [[0, 1.5], [0, 0]])


def test_invalid_window():
    with pytest.raises(ValueError):
        Corpus().fit_matrix([["a", "b"]], window_size=0)


def test_queries_exclude_self_even_when_scores_tie():
    model = GloVe(seed=0).fit_vectors(pair(), epochs=0)
    model.add_dictionary({"a": 0, "b": 1})
    model.word_vectors[:] = 1
    assert model.get_most_similar("b", number=1)[0][0] == "a"
    assert model.get_most_similar("missing") == []
    assert model.check_similarity("missing", "a") == -1.0
    with pytest.raises(KeyError):
        model.get_most_similar("missing", ignore_missing=False)


def test_query_validation_and_zero_vectors():
    assert cosine([0, 0], [1, 1]) == 0
    assert cosine([1, 0], [1, 0]) == 1
    with pytest.raises(ValueError):
        GloVe().get_most_similar("a")
    model = GloVe().fit_vectors(pair(), epochs=0)
    with pytest.raises(ValueError):
        model.add_dictionary({"a": 0})
    model.add_dictionary({"a": 0, "b": 1})
    with pytest.raises(ValueError):
        model.get_most_similar("a", number=-1)


def test_pickle_round_trip(tmp_path):
    corpus = Corpus(); corpus.fit_matrix([["a", "b", "a"]])
    corpus.save_file(tmp_path / "corpus.pkl")
    restored = Corpus.load_file(tmp_path / "corpus.pkl")
    assert restored.dictionary == corpus.dictionary
    np.testing.assert_array_equal(restored.matrix.toarray(), corpus.matrix.toarray())
    model = GloVe(seed=42).fit_vectors(corpus.matrix)
    model.add_dictionary(corpus.dictionary)
    model.save_file(tmp_path / "model.pkl")
    loaded = GloVe.load_file(tmp_path / "model.pkl")
    np.testing.assert_array_equal(loaded.word_vectors, model.word_vectors)
    assert loaded.get_most_similar("a") == model.get_most_similar("a")


def test_offline_token_helpers():
    assert lower_tokenizer("HELLO world!") == ["hello", "world", "!"]
    assert sp_remover(["this", "works"], {"this"}) == ["works"]
    assert word_stemmer(["running"]) == ["run"]
    assert prepare_data("Hello\nWorld", tokenizer=lower_tokenizer) == [["hello"], ["world"]]


def test_imports_have_no_downloads_or_file_writes(tmp_path):
    program = f'''import sys, socket
sys.path.insert(0, {str(ROOT)!r})
def blocked(*args, **kwargs): raise AssertionError("unexpected network access")
socket.create_connection = blocked
socket.socket.connect = blocked
import nltk
nltk.download = blocked
import Corpus, GloVe, lower_tokenizer, train_model
import color_docx, extend_docx
'''
    result = subprocess.run([sys.executable, "-c", program], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_fresh_cli_and_model_reload(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run([sys.executable, str(ROOT / "train_model.py"), "--epochs", "4", "--seed", "0", "--output", str(output)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["seed"] == 0 and report["vocabulary"] > 1
    assert report["final_objective"] < report["initial_objective"]
    loaded = GloVe.load_file(output / "model.pkl")
    assert loaded.dictionary and loaded.get_most_similar("research")
