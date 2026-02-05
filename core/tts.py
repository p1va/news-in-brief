import os
import wave
import base64
import subprocess
from typing import Optional, Protocol
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import save
from openai import OpenAI

load_dotenv()

class TTSProvider(Protocol):
    def generate(self, text: str, output_path: str) -> None:
        ...

class ElevenLabsProvider:
    def __init__(self, voice_id: str = "ZF6FPAbjXT4488VcRRnw", model_id: str = "eleven_v3"):
        self.voice_id = voice_id
        self.model_id = model_id
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
             # It seems the original code raised ValueError if key missing
             pass
        self.client = ElevenLabs(api_key=self.api_key)

    def generate(self, text: str, output_path: str) -> None:
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set.")
            
        print(f"Generating audio with ElevenLabs (voice_id={self.voice_id})...")
        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )
        save(audio, output_path)
        print(f"Audio saved to {output_path}")

class OpenRouterTTSProvider:
    def __init__(self, model: str = "openai/gpt-audio", voice: str = "marin"):
        self.model = model
        self.voice = voice
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def generate(self, text: str, output_path: str) -> None:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        print(f"Generating audio with OpenRouter (model={self.model})...")
        
        # OpenRouter/OpenAI Audio requires PCM16 streaming for text->audio
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Read the following news briefing script out loud: " + text}],
            modalities=["text", "audio"],
            audio={"voice": self.voice, "format": "pcm16"},
            stream=True
        )

        audio_buffer = bytearray()
        for chunk in completion:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'audio'):
                audio_chunk = chunk.choices[0].delta.audio
                if audio_chunk and 'data' in audio_chunk:
                    audio_buffer.extend(base64.b64decode(audio_chunk['data']))
        
        if len(audio_buffer) == 0:
            raise RuntimeError("No audio data received from OpenRouter.")

        # Save as temporary WAV
        temp_wav = str(Path(output_path).with_suffix(".temp.wav"))
        
        # 24kHz matches openai/gpt-audio-mini output
        with wave.open(temp_wav, "wb") as wf:
            wf.setnchannels(1) 
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(24000) 
            wf.writeframes(audio_buffer)
            
        # Convert to MP3 using ffmpeg
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_wav, "-b:a", "128k", output_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"Audio converted and saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg conversion failed: {e}")
            raise
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

class TextToSpeech:
    def __init__(
        self,
        voice_id: str = "marin",
        model_id: str = "openai/gpt-audio",
        provider: str = "openrouter"
    ):
        self.provider_name = provider
        
        if provider == "openrouter":
            # Map model_id to model, voice_id to voice
            self.provider = OpenRouterTTSProvider(model=model_id, voice=voice_id)
        else:
            self.provider = ElevenLabsProvider(voice_id=voice_id, model_id=model_id)

    def __call__(self, text: str, output_path: str, max_chars: int = 5000) -> None:
        """
        Generates audio from text using the configured provider.
        """
        if len(text) > max_chars:
            print(f"Warning: Text truncated from {len(text)} to {max_chars} characters")
            text = text[:max_chars]

        try:
            self.provider.generate(text, output_path)
        except Exception as e:
            print(f"Error generating audio: {e}")
            raise

if __name__ == "__main__":
    # Test run
    # To test OpenRouter: TextToSpeech(provider="openrouter")
    tts = TextToSpeech()
    tts(
        "Hello, this is a test of the Asia in Brief audio generation system.",
        "test_audio.mp3",
    )