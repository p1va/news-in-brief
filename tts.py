import os
from typing import Optional

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import save

load_dotenv()


class TextToSpeech:
    def __init__(
        self,
        voice_id: str = "ZF6FPAbjXT4488VcRRnw",  # Amelia
        model_id: str = "eleven_v3",
        api_key: Optional[str] = None,
    ):
        self.voice_id = voice_id
        self.model_id = model_id
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")

        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set.")

        self.client = ElevenLabs(api_key=self.api_key)

    def __call__(self, text: str, output_path: str) -> None:
        """
        Generates audio from text using ElevenLabs API and saves it to the specified path.
        """
        print(
            f"Generating audio with voice_id={self.voice_id} model_id={self.model_id}..."
        )
        try:
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format="mp3_44100_128",
            )
            save(audio, output_path)
            print(f"Audio saved to {output_path}")
        except Exception as e:
            print(f"Error generating audio: {e}")
            raise


if __name__ == "__main__":
    # Test run
    tts = TextToSpeech()
    tts(
        "Hello, this is a test of the Asia in Brief audio generation system.",
        "test_audio.mp3",
    )