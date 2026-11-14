"""Unit tests for the 8-byte binary audio frame packaging."""

import pytest

from joi.domain.events import pack_audio_frame, unpack_audio_frame


def test_pack_and_unpack_audio_frame() -> None:
    turn_id = 42
    seq_num = 7
    pcm_payload = b"\x00\x01\x02\x03\x04\x05"

    framed_data = pack_audio_frame(turn_id, seq_num, pcm_payload)
    assert len(framed_data) == 8 + len(pcm_payload)

    extracted_turn_id, extracted_seq_num, extracted_pcm = unpack_audio_frame(framed_data)
    assert extracted_turn_id == turn_id
    assert extracted_seq_num == seq_num
    assert extracted_pcm == pcm_payload


def test_unpack_audio_frame_too_short() -> None:
    short_data = b"\x00\x01\x02"
    with pytest.raises(ValueError, match="Binary frame too short"):
        unpack_audio_frame(short_data)
