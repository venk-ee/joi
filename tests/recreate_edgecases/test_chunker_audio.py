"""Demonstrates what bad sentence splitting sounds like vs proper sentence splitting."""

import asyncio
import numpy as np
import soundfile as sf
from joi.speech.tts import KokoroTTS

SAMPLE_RATE = 24000
PAUSE = np.zeros(int(SAMPLE_RATE * 0.35), dtype=np.int16)  # 350ms pause between naive chunks


async def generate_comparisons() -> None:
    tts = KokoroTTS()

    print("=== Generating Audio Comparison for Sentence Splitting ===\n")

    # -------------------------------------------------------------
    # 1. ABBREVIATION EDGE CASE: "Dr. Smith"
    # -------------------------------------------------------------
    print("1. Generating Abbreviation Case ('Dr. Smith')...")

    # BAD: Naively split at the period in "Dr."
    b1, _ = await tts.synthesize("Please call Dr.")
    b2, _ = await tts.synthesize("Smith immediately.")
    arr_b1 = np.frombuffer(b1, dtype=np.int16)
    arr_b2 = np.frombuffer(b2, dtype=np.int16)
    bad_abbrev = np.concatenate([arr_b1, PAUSE, arr_b2])
    sf.write("bad_abbreviation.wav", bad_abbrev, SAMPLE_RATE)
    print("   -> Created bad_abbreviation.wav (Listen to the weird cadence drop after 'Dr.')")

    # GOOD: Kept together
    g1, _ = await tts.synthesize("Please call Dr. Smith immediately.")
    good_abbrev = np.frombuffer(g1, dtype=np.int16)
    sf.write("good_abbreviation.wav", good_abbrev, SAMPLE_RATE)
    print("   -> Created good_abbreviation.wav (Natural title and name flow)")

    # -------------------------------------------------------------
    # 2. DECIMAL EDGE CASE: "3.14"
    # -------------------------------------------------------------
    print("\n2. Generating Decimal Case ('3.14')...")

    # BAD: Naively split at the decimal dot "3."
    d1, _ = await tts.synthesize("The sensor recorded 3.")
    d2, _ = await tts.synthesize("14 volts on the circuit.")
    arr_d1 = np.frombuffer(d1, dtype=np.int16)
    arr_d2 = np.frombuffer(d2, dtype=np.int16)
    bad_decimal = np.concatenate([arr_d1, PAUSE, arr_d2])
    sf.write("bad_decimal.wav", bad_decimal, SAMPLE_RATE)
    print("   -> Created bad_decimal.wav (Listen to 'three... fourteen volts')")

    # GOOD: Kept together
    gd, _ = await tts.synthesize("The sensor recorded 3.14 volts on the circuit.")
    good_decimal = np.frombuffer(gd, dtype=np.int16)
    sf.write("good_decimal.wav", good_decimal, SAMPLE_RATE)
    print("   -> Created good_decimal.wav (Proper 'three point one four' cadence)")

    print("\n=== All audio samples generated! ===")


if __name__ == "__main__":
    asyncio.run(generate_comparisons())
