"""Train from a fresh checkout without saved dictionaries or downloaded corpora."""
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--query", default="research")
    args = parser.parse_args(argv)
    try:
        corpus = Corpus()
        corpus.fit_matrix(prepare_data(args.corpus.read_text(encoding="utf-8"), tokenizer=lower_tokenizer), args.window)
        model = GloVe(dim=args.dim, seed=args.seed)
        model.fit_vectors(corpus.matrix, epochs=args.epochs)
        model.add_dictionary(corpus.dictionary)
        args.output.mkdir(parents=True, exist_ok=True)
        corpus.save_file(args.output / "corpus.pkl")
        model.save_file(args.output / "model.pkl")
        report = {
            "seed": args.seed, "epochs": args.epochs, "vocabulary": len(corpus.dictionary),
            "cooccurrence_pairs": int(corpus.matrix.nnz), "initial_objective": model.loss_history[0],
            "final_objective": model.loss_history[-1], "query": args.query,
            "neighbors": model.get_most_similar(args.query),
            "note": "Toy demonstration, not a validated semantic-quality benchmark. Pickles are trusted-local artifacts only."
        }
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, FloatingPointError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
