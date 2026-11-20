import asyncio

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

WISPER_SAMPLE_RATE = 16000
FLOAT32_MAX = 32768.0

class WhisperSTT:
    def __init__(self, model_size: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        processed_audio = self._process_audio(audio_data, sample_rate)
        text = await asyncio.to_thread(self._run_inference, processed_audio)
        return text

    def _run_inference(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, beam_size=1, language="en", vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _process_audio(self, audio_data: bytes, sample_rate: int) -> np.ndarray:
        
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        
        # Convert to float32 and normalize to [-1, 1]
        audio_float = audio_int16.astype(np.float32) / FLOAT32_MAX
        
        # Resample to 16kHz if needed
        if sample_rate != WISPER_SAMPLE_RATE:
            target_len = int(len(audio_float) * WISPER_SAMPLE_RATE / sample_rate)
            original_time_point = np.linspace(0, len(audio_float), len(audio_float))
            new_time_point = np.linspace(0, len(audio_float), target_len)
            processed_audio_float = np.interp(new_time_point, original_time_point, audio_float).astype(np.float32)
            
        else:
            processed_audio_float = audio_float.astype(np.float32)
        
        return processed_audio_float


def main():
    stt = WhisperSTT()
    print("STT model loaded")
    audio_file="output.wav"
    audio_data, sample_rate = sf.read(audio_file, dtype='int16')
    text = asyncio.run(stt.transcribe(audio_data.tobytes(), sample_rate=sample_rate))
    print(f"Audio length: {len(audio_data)} samples")
    print(f"Sample rate: {sample_rate}")
    print(f"Transcribed text: {text}")


if __name__ == "__main__":
    main()