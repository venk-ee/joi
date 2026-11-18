"""Automated unit test suite for SentenceChunker streaming and edge cases."""

from joi.parser.chunker import SentenceChunker


def test_basic_sentence_splitting() -> None:
    chunker = SentenceChunker()
    sentences = chunker.feed("Hello world! How are you doing today? ")
    assert sentences == ["Hello world!", "How are you doing today?"]
    assert chunker.flush() is None


def test_abbreviation_preservation() -> None:
    chunker = SentenceChunker()
    # "Dr." should not cause a split
    sentences = chunker.feed("Please call Dr. Smith immediately. ")
    assert sentences == ["Please call Dr. Smith immediately."]


def test_multiple_abbreviations_in_one_sentence() -> None:
    chunker = SentenceChunker()
    sentences = chunker.feed("Mr. Smith and Dr. Watson arrived at the station. It was dark. ")
    assert sentences == [
        "Mr. Smith and Dr. Watson arrived at the station.",
        "It was dark.",
    ]


def test_decimal_preservation() -> None:
    chunker = SentenceChunker()
    sentences = chunker.feed("The item costs $10.50 on sale. That is a great deal! ")
    assert sentences == [
        "The item costs $10.50 on sale.",
        "That is a great deal!",
    ]


def test_streaming_token_by_token() -> None:
    chunker = SentenceChunker()
    tokens = ["The ", "value ", "of ", "pi ", "is ", "3.", "14 ", "roughly. ", "Awesome! "]

    collected: list[str] = []
    for token in tokens:
        collected.extend(chunker.feed(token))

    assert collected == [
        "The value of pi is 3.14 roughly.",
        "Awesome!",
    ]


def test_flush_handles_trailing_text_without_punctuation() -> None:
    chunker = SentenceChunker()
    chunker.feed("This text has no punctuation at all")
    flushed = chunker.flush()
    assert flushed == "This text has no punctuation at all"
