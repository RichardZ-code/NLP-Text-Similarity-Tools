"""Train a small reproducible model from a text file, one context per line."""
import argparse
from pathlib import Path

from Corpus import Corpus, prepare_data
from GloVe import GloVe
from lower_tokenizer import lower_tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).parent / "examples/corpus.txt")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--query", default="research")
    args = parser.parse_args(argv)
    try:
        texts = args.corpus.read_text(encoding="utf-8")
        corpus = Corpus()
        corpus.fit_matrix(prepare_data(texts, tokenizer=lower_tokenizer), window_size=args.window)
        model = GloVe(dim=args.dim, seed=args.seed)
        model.fit_vectors(corpus.matrix, epochs=args.epochs)
        model.add_dictionary(corpus.dictionary)
        args.output.mkdir(parents=True, exist_ok=True)
        corpus.save_file(args.output / "corpus.pkl")
        model.save_file(args.output / "model.pkl")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Trained {len(corpus.dictionary)} words across {corpus.matrix.nnz} co-occurrence pairs; seed={args.seed}")
    print(f"Saved trusted-local-use pickle files to {args.output}")
    for word, score in model.get_most_similar(args.query):
        print(f"{word}\t{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
