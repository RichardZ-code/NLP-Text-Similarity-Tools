"""Compatibility entry point for the bundled, tested text-similarity demo.

The original Python 2 Doc2Vec scratch script remains in Git history.
"""
from train_model import main


if __name__ == "__main__":
    raise SystemExit(main())
