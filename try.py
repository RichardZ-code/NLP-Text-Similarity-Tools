"""Optional Doc2Vec demonstration, separate from the tested GloVe pipeline.

Requires gensim. Uses the same synthetic corpus as the main example and does
not download data or execute training when imported.
"""
from pathlib import Path


def main():
    try:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        from gensim.utils import simple_preprocess
    except ImportError as error:
        raise SystemExit("This optional example needs gensim: python -m pip install gensim") from error

    corpus_path = Path(__file__).parent / "examples" / "corpus.txt"
    sentences = [line for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    documents = [TaggedDocument(simple_preprocess(line), [index]) for index, line in enumerate(sentences)]
    model = Doc2Vec(vector_size=20, min_count=1, epochs=30, workers=1, seed=42)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    query = "research teams compare experiment results"
    vector = model.infer_vector(simple_preprocess(query))
    print(f"Query: {query}")
    print("Toy similarities, not a validated retrieval benchmark:")
    for index, score in model.dv.most_similar([vector], topn=3):
        print(f"{score:.4f}\t{sentences[index]}")


if __name__ == "__main__":
    main()
