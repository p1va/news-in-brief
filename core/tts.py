import base64
import mimetypes
import os
import struct
import subprocess
import wave
from pathlib import Path
from typing import Protocol

import imageio_ffmpeg as iio_ffmpeg
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import save
from openai import OpenAI
from google import genai
from google.genai import types

load_dotenv()


class TTSProvider(Protocol):
    def generate(self, text: str, output_path: str) -> None: ...


class ElevenLabsProvider:
    def __init__(
        self, voice_id: str = "ZF6FPAbjXT4488VcRRnw", model_id: str = "eleven_v3"
    ):
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
            messages=[
                {
                    "role": "user",
                    "content": "Read the following news briefing script out loud: "
                    + text,
                }
            ],
            modalities=["text", "audio"],
            audio={"voice": self.voice, "format": "pcm16"},
            stream=True,
        )

        audio_buffer = bytearray()
        for chunk in completion:
            if hasattr(chunk.choices[0], "delta") and hasattr(
                chunk.choices[0].delta, "audio"
            ):
                audio_chunk = chunk.choices[0].delta.audio
                if audio_chunk and "data" in audio_chunk:
                    audio_buffer.extend(base64.b64decode(audio_chunk["data"]))

        if len(audio_buffer) == 0:
            raise RuntimeError("No audio data received from OpenRouter.")

        # Save as temporary WAV
        temp_wav = Path(output_path).with_suffix(".temp.wav")

        # 24kHz matches openai/gpt-audio-mini output
        with wave.open(str(temp_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(24000)
            wf.writeframes(audio_buffer)

        # Convert to MP3 using ffmpeg
        try:
            ffmpeg_exe = iio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", str(temp_wav), "-b:a", "128k", output_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f"Audio converted and saved to {output_path}")
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"FFmpeg conversion failed: {e}")
            raise
        finally:
            temp_wav.unlink(missing_ok=True)


class GeminiTTSProvider:
    def __init__(self, voice_id: str = "Despina", model: str = "gemini-2.5-pro-preview-tts"):
        self.voice_name = voice_id
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            pass
        self.client = genai.Client(api_key=self.api_key)

    def _convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        parameters = self._parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"]
        sample_rate = parameters["rate"]
        num_channels = 1
        data_size = len(audio_data)
        bytes_per_sample = bits_per_sample // 8
        block_align = num_channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        chunk_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            chunk_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size
        )
        return header + audio_data

    def _parse_audio_mime_type(self, mime_type: str) -> dict[str, int]:
        bits_per_sample = 16
        rate = 24000
        parts = mime_type.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate_str = param.split("=", 1)[1]
                    rate = int(rate_str)
                except (ValueError, IndexError):
                    pass
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass
        return {"bits_per_sample": bits_per_sample, "rate": rate}

    def generate(self, text: str, output_path: str) -> None:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        print(f"Generating audio with Gemini (model={self.model}, voice={self.voice_name})...")

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="Read the following news briefing script out loud: " + text),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice_name
                    )
                )
            ),
        )

        audio_buffer = bytearray()
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.parts is None:
                continue
            for part in chunk.parts:
                if part.inline_data and part.inline_data.data:
                    # Append the raw audio chunk
                    audio_buffer.extend(part.inline_data.data)

        if len(audio_buffer) == 0:
            raise RuntimeError("No audio data received from Gemini.")

        # Gemini returns audio chunks with mime type, usually audio/L16;rate=24000
        # Since we just appended all chunks together, we need to add the wav header once at the beginning
        # We will assume a standard format unless we parse the first chunk, but we can just use the
        # hardcoded defaults from _parse_audio_mime_type (16-bit, 24kHz)
        mime_type = "audio/L16;rate=24000" 
        if chunk.parts and chunk.parts[0].inline_data and chunk.parts[0].inline_data.mime_type:
            mime_type = chunk.parts[0].inline_data.mime_type
            
        wav_data = self._convert_to_wav(bytes(audio_buffer), mime_type)

        temp_wav = Path(output_path).with_suffix(".temp.wav")
        with open(temp_wav, "wb") as f:
            f.write(wav_data)

        # Convert to MP3 using ffmpeg
        try:
            ffmpeg_exe = iio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", str(temp_wav), "-b:a", "128k", output_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f"Audio converted and saved to {output_path}")
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"FFmpeg conversion failed: {e}")
            raise
        finally:
            temp_wav.unlink(missing_ok=True)


class TextToSpeech:
    def __init__(
        self,
        voice_id: str = "marin",
        model_id: str = "openai/gpt-audio",
        provider: str = "openrouter",
    ):
        self.provider_name = provider

        if provider == "openrouter":
            self.provider = OpenRouterTTSProvider(model=model_id, voice=voice_id)
        elif provider == "gemini":
            self.provider = GeminiTTSProvider(voice_id=voice_id, model=model_id)
        else:
            self.provider = ElevenLabsProvider(voice_id=voice_id, model_id=model_id)

    def __call__(self, text: str, output_path: str, max_chars: int = 7000) -> None:
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
    tts = TextToSpeech()
    tts(
        "Hello, this is a test of the Asia in Brief audio generation system.",
        "test_audio.mp3",
    )
