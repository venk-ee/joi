"""Streaming token parser with inline tag extraction, boundary protection, and sentence chunking."""

import re
from collections.abc import AsyncGenerator

from joi.domain.models import SentenceChunk, TagCategory, TagEvent

# Known vocabulary sets
EXPRESSION_TAGS = {"smile", "curious", "thoughtful", "smirk", "concerned", "giggle", "neutral"}
ACTION_TAGS = {"sit", "stand", "lean_forward", "look_away", "wave", "nod"}

MIN_SENTENCE_LENGTH = 4
MAX_TAG_LENGTH = 24  # Enforces bounded tag buffer to prevent runaway allocations
ABBREVIATIONS = {"dr.", "mr.", "mrs.", "ms.", "e.g.", "i.e.", "vs."}


class StreamingParser:
    """Consumes an asynchronous token stream from an LLM.

    Extracts inline [tag] events in real time and buffers clean text until sentence
    boundaries are hit, yielding SentenceChunk objects for low-latency TTS synthesis.
    Protects against decimal splits (e.g. 3.14) and runaway bracket buffers.
    """

    def __init__(self, turn_id: int) -> None:
        self.turn_id = turn_id
        self._tag_buffer = ""
        self._text_buffer = ""
        self._in_tag = False
        self._sentence_idx = 0

    async def parse_stream(
        self,
        token_stream: AsyncGenerator[str, None],
    ) -> AsyncGenerator[TagEvent | SentenceChunk, None]:
        """Processes incoming tokens chunk-by-chunk."""
        async for token in token_stream:
            for char in token:
                if self._in_tag:
                    self._tag_buffer += char
                    if char == "]":
                        # End of tag detected
                        tag_event = self._extract_tag(self._tag_buffer)
                        if tag_event:
                            yield tag_event
                        else:
                            # Not a recognized tag (e.g. [1, 2, 3]), return to text buffer
                            self._text_buffer += self._tag_buffer
                        self._tag_buffer = ""
                        self._in_tag = False
                    elif len(self._tag_buffer) > MAX_TAG_LENGTH:
                        # Abort tag parsing: runaway bracket detected
                        self._text_buffer += self._tag_buffer
                        self._tag_buffer = ""
                        self._in_tag = False
                elif char == "[":
                    self._in_tag = True
                    self._tag_buffer = "["
                else:
                    self._text_buffer += char

                    # Check for sentence boundary with decimal/abbreviation protection
                    if self._is_sentence_boundary(char):
                        sentence = self._flush_text_buffer()
                        if sentence:
                            yield SentenceChunk(
                                text=sentence,
                                sentence_idx=self._sentence_idx,
                                turn_id=self.turn_id,
                            )
                            self._sentence_idx += 1

        # Flush any remaining unclosed tag as raw text
        if self._tag_buffer:
            self._text_buffer += self._tag_buffer
            self._tag_buffer = ""

        # Flush final remaining text
        final_sentence = self._flush_text_buffer()
        if final_sentence:
            yield SentenceChunk(
                text=final_sentence,
                sentence_idx=self._sentence_idx,
                turn_id=self.turn_id,
            )
            self._sentence_idx += 1

    def _is_sentence_boundary(self, char: str) -> bool:
        """Determines if the character represents a true spoken sentence termination."""
        stripped = self._text_buffer.strip()
        if len(stripped) < MIN_SENTENCE_LENGTH:
            return False

        if char in {"!", "?", "\n"}:
            return True

        if char == ".":
            # Protect against decimal numbers (e.g. "3.14" -> preceding char is a digit)
            buf = self._text_buffer
            if len(buf) >= 2 and buf[-2].isdigit():
                return False

            # Protect against common abbreviations (e.g. "Dr.", "e.g.")
            last_word = stripped.split()[-1].lower() if stripped.split() else ""
            return last_word not in ABBREVIATIONS

        return False

    def _flush_text_buffer(self) -> str:
        """Cleans and returns the current text buffer."""
        cleaned = self._clean_text(self._text_buffer)
        self._text_buffer = ""
        return cleaned

    def _extract_tag(self, raw_tag: str) -> TagEvent | None:
        """Parses a string like '[smile]' or '[sit]' into a TagEvent."""
        match = re.match(r"^\[([a-zA-Z0-9_]+)(?::([0-9.]+))?\]$", raw_tag.strip())
        if not match:
            return None

        tag_name = match.group(1).lower()
        param_val = match.group(2)

        parameters: dict[str, float | str] = {}
        if param_val is not None:
            try:
                parameters["intensity"] = float(param_val)
            except ValueError:
                parameters["value"] = param_val

        if tag_name in ACTION_TAGS:
            return TagEvent(name=tag_name, category=TagCategory.ACTION, parameters=parameters)
        if tag_name in EXPRESSION_TAGS:
            return TagEvent(name=tag_name, category=TagCategory.EXPRESSION, parameters=parameters)

        return TagEvent(name=tag_name, category=TagCategory.EXPRESSION, parameters=parameters)

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """Strips excessive whitespace and dangling artifacts before passing to TTS."""
        cleaned = re.sub(r"\s+", " ", raw_text).strip()
        return cleaned
