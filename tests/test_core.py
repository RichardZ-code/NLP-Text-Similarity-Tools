import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from Corpus import Corpus, prepare_data
from GloVe import GloVe, cosine
from lower_tokenizer import lower_tokenizer, sp_remover, word_stemmer

ROOT = Path(__file__).resolve().parents[1]


def pair(count=2.0):
    return coo_matrix(([count], ([0], [1])), shape=(2, 2))


def trained():
    model = GloVe(seed=0).fit_vectors(pair(), epochs=2)
    model.add_dictionary({"alpha": 0, "beta": 1})
    return model


def test_biases_both_receive_the_same_gradient():
    model = GloVe(seed=0).fit_vectors(pair(), epochs=1)
    assert model.word_biases[0] != 0
    assert model.word_biases[0] == model.word_biases[1]
    np.testing.assert_array_equal(model.biases_sum_gradients[:1], model.biases_sum_gradients[1:])


def test_both_vector_gradients_use_old_parameters():
    model = GloVe(dim=2, learning_rate=0.1, max_count=1, seed=0)
    original = (np.random.RandomState(0).rand(2, 2) - 0.5) / 2
    gradient = np.dot(original[0], original[1]) - np.log(2)
    expected = original.copy()
    expected[0] -= 0.1 * gradient * original[1]
    expected[1] -= 0.1 * gradient * original[0]
    model.fit_vectors(pair(), epochs=1)
    np.testing.assert_allclose(model.word_vectors, expected, rtol=1e-13, atol=1e-13)


def test_gradient_matches_independent_finite_difference():
    model = GloVe(dim=2, learning_rate=0.01, max_count=10, seed=0)
    original = np.array([[0.2, -0.3], [0.4, 0.5]])
    biases = np.array([0.1, -0.2])
    weight = (3.0 / 10) ** 0.5
    parameters = np.concatenate([original.ravel(), biases])

    def loss(p):
        vectors = p[:4].reshape(2, 2)
        residual = vectors[0] @ vectors[1] + p[4] + p[5] - np.log(3)
        return 0.5 * weight * residual**2

    eps = 1e-6
    gradient = []
    for index in range(6):
        offset = np.zeros(6); offset[index] = eps
        gradient.append((loss(parameters + offset) - loss(parameters - offset)) / (2 * eps))
    model.word_vectors, model.word_biases = original.copy(), biases.copy()
    model.vectors_sum_gradients = np.ones_like(original)
    model.biases_sum_gradients = np.ones(2)
    model._step(0, 1, 3.0)
    actual = np.concatenate([model.word_vectors.ravel(), model.word_biases])
    np.testing.assert_allclose(actual, parameters - 0.01 * np.array(gradient), atol=1e-10)


@pytest.mark.parametrize("seed", [0, 17])
def test_seed_is_reproducible(seed):
    a = GloVe(seed=seed).fit_vectors(pair(), epochs=3)
    b = GloVe(seed=seed).fit_vectors(pair(), epochs=3)
    np.testing.assert_array_equal(a.word_vectors, b.word_vectors)
    assert a.loss_history == b.loss_history


def test_loss_decreases_on_small_fixture():
    model = GloVe(seed=1, learning_rate=0.05).fit_vectors(pair(3), epochs=100)
    assert np.isfinite(model.loss_history).all()
    assert model.loss_history[-1] < model.loss_history[0]


@pytest.mark.parametrize("matrix", [coo_matrix((0, 0)), pair(-1), pair(float("nan")),
                                   coo_matrix(([1], ([0], [0])), shape=(2, 2))])
def test_invalid_training_matrices_are_rejected(matrix):
    with pytest.raises(ValueError):
        GloVe().fit_vectors(matrix)


@pytest.mark.parametrize("kwargs", [{"dim": 0}, {"seed": -1}, {"learning_rate": 0}, {"alpha": float("inf")}])
def test_invalid_hyperparameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GloVe(**kwargs)


def test_corpus_instances_do_not_share_dictionary():
    first, second = Corpus(), Corpus()
    first.fit_matrix([["alpha", "beta"]])
    assert second.dictionary == {}


def test_cooccurrence_distance_weighting():
    corpus = Corpus()
    corpus.fit_matrix([["a", "b", "c"]], window_size=2)
    np.testing.assert_array_equal(corpus.matrix.toarray(), [[0, 1, 0.5], [0, 0, 1], [0, 0, 0]])
    assert corpus.dictionary == {"a": 0, "b": 1, "c": 2}


def test_tokenizer_and_supplied_stopwords_work_offline():
    assert lower_tokenizer("DATA pipelines.") == ["data", "pipelines", "."]
    assert sp_remover(["the", "data"], stop_words=["the"]) == ["data"]
    assert word_stemmer(["running"]) == ["run"]
    assert prepare_data("a b\nc d") == [["a", "b"], ["c", "d"]]


def test_imports_do_not_download_or_print(monkeypatch, capsys):
    import nltk
    def forbidden(*args, **kwargs):
        raise AssertionError("implicit download")
    monkeypatch.setattr(nltk, "download", forbidden)
    for name in ("Corpus", "lower_tokenizer", "train_model"):
        importlib.reload(importlib.import_module(name))
    assert capsys.readouterr().out == ""


def test_similarity_excludes_query_even_when_vectors_tie():
    model = trained()
    model.word_vectors[:] = 1
    matches = model.get_most_similar("beta")
    assert len(matches) == 1 and matches[0][0] == "alpha"
    assert matches[0][1] == pytest.approx(1.0)
    assert model.get_most_similar("beta", number=0) == []


def test_unknown_words_and_zero_vectors():
    model = trained()
    assert model.get_most_similar("missing") == []
    with pytest.raises(KeyError):
        model.get_most_similar("missing", ignore_missing=False)
    assert cosine([0, 0], [1, 2]) == 0
    assert model.check_similarity("alpha", "alpha") == pytest.approx(1)


def test_pickle_round_trip_for_trusted_local_files(tmp_path):
    model = trained()
    path = tmp_path / "model.pkl"
    model.save_file(path)
    restored = GloVe.load_file(path)
    np.testing.assert_array_equal(model.word_vectors, restored.word_vectors)
    assert model.get_most_similar("alpha") == restored.get_most_similar("alpha")


def test_complete_cli_from_another_directory(tmp_path):
    reports = []
    for number in range(2):
        output = tmp_path / str(number)
        result = subprocess.run([sys.executable, str(ROOT / "train_model.py"),
                                 "--output", str(output), "--epochs", "5"],
                                cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        reports.append(json.loads((output / "report.json").read_text()))
        assert (output / "model.pkl").exists() and (output / "corpus.pkl").exists()
    assert reports[0] == reports[1]


def test_cli_reports_missing_input(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "train_model.py"),
                             "--corpus", str(tmp_path / "missing.txt")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_compatibility_demo_entrypoint(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "try.py"), "--epochs", "2",
                             "--output", str(tmp_path / "demo")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "demo" / "report.json").exists()
