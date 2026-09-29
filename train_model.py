"""Train the bundled toy corpus from scratch; no pretrained files required."""
import argparse
import json
from pathlib import Path

from Corpus import Corpus, prepare_data
from GloVe import GloVe
from lower_tokenizer import lower_tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).parent / "examples" / "corpus.txt")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--query", default="data")
    args = parser.parse_args(argv)
    try:
        text = args.corpus.read_text(encoding="utf-8")
        corpus = Corpus()
        corpus.fit_matrix(prepare_data(text, tokenizer=lower_tokenizer))
        model = GloVe(dim=args.dim, seed=args.seed)
        model.fit_vectors(corpus.matrix, epochs=args.epochs)
        model.add_dictionary(corpus.dictionary)
        neighbors = model.get_most_similar(args.query, ignore_missing=False)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    corpus.save_file(args.output / "corpus.pkl")
    model.save_file(args.output / "model.pkl")
    report = {"seed": args.seed, "vocabulary_size": len(corpus.dictionary),
              "cooccurrence_pairs": corpus.matrix.nnz, "epochs": args.epochs,
              "initial_loss": model.loss_history[0], "final_loss": model.loss_history[-1],
              "query": args.query, "neighbors": neighbors,
              "note": "Toy corpus demonstration, not an accuracy benchmark."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
