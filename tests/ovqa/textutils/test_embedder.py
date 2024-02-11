import pytest

from ovqa.textutils.embeddings import (
    get_sentence_embedder,
    EmbeddingsPackageConst,
    normalize_embeddings,
)

sentences = [
    "hello world",
    "goodbye world",
    "beans",
    "peas",
]

too_long_sentence = "a sentence that is way too long " * 100


@pytest.mark.slow
def test_embedder_simple():
    ember = get_sentence_embedder(
        model_name="all-MiniLM-L6-v2",
        package_name=EmbeddingsPackageConst.SENTENCE_TRANSFORMERS,
    )
    outputs = ember.encode(sentences)
    outputs = normalize_embeddings(outputs)
    similarities = outputs @ outputs.T
    for n_src, text_src in enumerate(sentences):
        for n_tgt in range(n_src + 1, len(sentences)):
            text_tgt = sentences[n_tgt]
            print(f"{text_src:>20s} to {text_tgt:<20s}: {similarities[n_src, n_tgt]:.3f}")

    ember.encode([too_long_sentence])


@pytest.mark.slow
def test_embedder_db():
    ember = get_sentence_embedder(
        model_name="all-MiniLM-L6-v2",
        package_name=EmbeddingsPackageConst.SENTENCE_TRANSFORMERS,
        use_db=True,
    )
    outputs = ember.encode(sentences)
    outputs = normalize_embeddings(outputs)
    similarities = outputs @ outputs.T
    for n_src, text_src in enumerate(sentences):
        for n_tgt in range(n_src + 1, len(sentences)):
            text_tgt = sentences[n_tgt]
            print(f"{text_src:>20s} to {text_tgt:<20s}: {similarities[n_src, n_tgt]:.3f}")

    ember.encode([too_long_sentence])


def main():
    test_embedder_simple()
    test_embedder_db()


if __name__ == "__main__":
    main()
