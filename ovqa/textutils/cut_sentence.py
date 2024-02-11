from nltk import word_tokenize

from visiontext.nlp import (
    tokenize_sentences_and_connectors,
    tokenize_words_and_connectors,
    ensure_setup_nltk,
    rebuild_from_words_and_connectors,
)


def cut_too_long_text(
    input_str,
    verbose=False,
    max_words_soft=40,
    max_words_hard=50,
    language="english",
    long_answer=True,
    short_answer=True,
):
    """
    If the input has less than M_hard words, return the input. Otherwise:
    If possible cut at sentence boundaries s.t. the output has between M_soft and M_hard words.
    If not possible, cut in the middle of sentence at M_soft + (M_hard - M_soft) / 2 words.

    Note: This can still lead to very long outputs / very many tokens if there are very long
    words (sequences without spaces) in the input.
    """
    assert max_words_soft <= max_words_hard
    ensure_setup_nltk()

    # if the input is a long answer, we want to cut it before this text
    if long_answer:
        # find "Long answer:" and cut before it
        long_answer_idx = input_str.lower().find("long answer:")
        if long_answer_idx != -1:
            input_str = input_str[:long_answer_idx]

    if short_answer:
        # find "Short answer:" and cut after it
        short_answer_idx = input_str.lower().find("short answer:")
        if short_answer_idx != -1:
            input_str = input_str[:short_answer_idx]

    input_str = input_str.strip()
    if len(input_str) < max_words_hard:
        # if the string has less than N characters, it also must have less than N words.
        return input_str

    words = word_tokenize(input_str, language="english")
    if len(words) <= max_words_hard:
        # if the string has less than N words, everything is OK
        return input_str

    # at this point the input has more than the max N words
    # tokenize s.t. the spaces are preserved, in order to later reconstruct the str
    output_words_conns = []
    sentences_conns = tokenize_sentences_and_connectors(input_str, language="english")
    for sent, sent_conn in sentences_conns:
        words_conns_here = tokenize_words_and_connectors(sent, language=language, split_eos=False)

        # add the next sentence to the total output
        output_words_conns.extend(words_conns_here)
        output_words_conns[-1][1] += sent_conn  # also add the space connecting the 2 sentences
        new_len = len(output_words_conns)

        if max_words_hard > new_len >= max_words_soft:
            # at this point the text has a length between min and max, return it
            if verbose:
                print(f"Cut at sentence boundary with output length {len(output_words_conns)}")
            return rebuild_from_words_and_connectors(output_words_conns).strip()

        if new_len >= max_words_hard:
            # before adding this sentence the text was too short, now its too long, so we have to
            # cut in the middle of the sentence.
            # to create a statistically fair comparison between cutting inside sentences and cutting
            # at sentence boundaries, cut at max_words_soft + (max_words_hard - max_words_soft) / 2
            # this way on average the output length of these 2 cases will be the same
            cut_point = int(max_words_soft + (max_words_hard - max_words_soft) // 2)
            new_words_conns = output_words_conns[:cut_point]
            if verbose:
                print(f"Cut at middle of sentence with output length {len(new_words_conns)}")
            return rebuild_from_words_and_connectors(new_words_conns).strip()

    raise RuntimeError(f"Cutting too long text failed, this should not happen.")


def main():
    for short_text in (
        "snake",
        "The image features a large dark snake, possibly a cobra.",
    ):
        print(f"{len(short_text):2d} {short_text}")
        processed_text = cut_too_long_text(short_text, verbose=True)
        assert processed_text == short_text

    # example that will be cut at sentence boundary
    example_text = (
        "The image features a large dark snake, possibly a cobra, lying on the sandy ground. "
        "It appears to be resting or sunning itself in the open area. As a language model, "
        "I cannot provide any specific details. I cannot provide details "
        "about the snake's species or location without more information. "
    )
    print(f"Long input: {example_text}")
    print()
    out = cut_too_long_text(example_text, verbose=True)
    print(out)
    print()

    # example that will be cut at middle of sentence
    example_text = (
        "The image features a large dark snake, possibly a cobra, lying on the sandy ground. "
        "It appears to be resting or sunning itself in the open area, but as a language model, "
        "I cannot provide any specific details about the snake's species or location without "
        "more information. "
    )
    print(f"Long input: {example_text}")
    print()
    out = cut_too_long_text(example_text, verbose=True)
    print(out)


if __name__ == "__main__":
    main()
