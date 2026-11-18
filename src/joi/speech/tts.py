import kokoro_onnx
import os
import asyncio
import numpy as np
import soundfile as sf

KOKORO_MODELS_DIR = "models"
KOKORO_MODEL_FILE = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = "voices-v1.0.bin"
KOKORO_SAMPLE_RATE = 24000

class KokoroTTS:
    def __init__(self):
        self._load_model()

    async def synthesize(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> tuple[bytes, int]:
        return await asyncio.to_thread(self._run_inference, text, voice, speed)

    def _load_model(self):
        model_path = os.path.join(KOKORO_MODELS_DIR, KOKORO_MODEL_FILE)
        voices_path = os.path.join(KOKORO_MODELS_DIR, KOKORO_VOICES_FILE)
        self.model = kokoro_onnx.Kokoro(model_path, voices_path)

    def _run_inference(self, text: str, voice: str, speed: float) -> tuple[bytes, int]:
        sample, sample_rate = self.model.create(text, voice, speed)

        pulse_modulation = (sample * 32767).clip(-32768, 32767).astype(np.int16)
        
        return pulse_modulation.tobytes(), KOKORO_SAMPLE_RATE



async def main():
    tts = KokoroTTS()
    text = "Hello! I am Joi, your personal AI companion.!"
    audio, sample_rate = await tts.synthesize(text)
    audio_data = np.frombuffer(audio, dtype=np.int16)
    sf.write("output.wav", audio_data, sample_rate)
    print("Success! Created output.wav")

if __name__ == "__main__":
    asyncio.run(main())