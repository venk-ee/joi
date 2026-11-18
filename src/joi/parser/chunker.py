


import re

COMMON_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "vs.", "etc.", "e.g.", "i.e.", "st.", "ave.", "gen.",
}
COMMON_ABBREVIATIONS_REGEX = re.compile(r"\b(" + "|".join(COMMON_ABBREVIATIONS) + r")\b")

PUNCTUATION_REGEX = re.compile(r"([.!?\n]+(?:\s+))")
DECIMAL_REGEX = re.compile(r"\b\d+\.\d+\b")

class SentenceChunker:
    def __init__(self) -> None:
        self._buffer: str = ""

    def feed(self, token: str) -> list[str]:
        self._buffer += token
        sentences: list[str] = []
        start_pos = 0

        while True:
            match = PUNCTUATION_REGEX.search(self._buffer, start_pos)
            if not match:
                break

            candidate = self._buffer[:match.end()].strip()
            if self._is_false_boundary(candidate):
                start_pos = match.end()
                continue
            if self._has_words(candidate):
                sentences.append(candidate)
            
            self._buffer = self._buffer[match.end():].lstrip()

            start_pos = 0
            
        return sentences

    def flush(self) -> str | None:
        remaining = self._buffer.strip()
        self._buffer = ""
        
        return remaining if self._has_words(remaining) else None

    def _is_false_boundary(self, candidate: str) -> bool:
        """Checks if the punctuation belongs to a decimal or an
        abbreviation."""
        return self._is_decimal(candidate) or self._is_abbreviation(candidate)

    def _is_decimal(self, candidate: str) -> bool:
        return bool(re.search(r"\d+\.\d+$", candidate))

    def _is_abbreviation(self, candidate: str) -> bool:
        words = candidate.lower().split()
        return bool(words and words[-1] in COMMON_ABBREVIATIONS)

    def _has_words(self, text: str) -> bool:
        """Returns True if text contains at least one alphanumeric
        character."""
        return bool(re.search(r"[a-zA-Z0-9]", text))

if __name__ == "__main__":
    chunker = SentenceChunker()
    print(chunker.feed("Hello, world!"))
    print(chunker.feed("This is a test."))
    print(chunker.flush())
    print(chunker.feed("Mr. Smith went to the store."))
    print(chunker.flush())
    print(chunker.feed("The price is $10.50."))
    print(chunker.flush())