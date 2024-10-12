#%%
from google.cloud import texttospeech
import io
import random
import base64

class TextToSpeech:
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=1
        )
        self.voices = [
            "en-GB-Journey-D",
            "en-GB-News-K",
            "en-GB-Wavenet-A",
            "en-GB-Wavenet-F"
        ]

    def text_to_speech(self, text):
        voice_name = random.choice(self.voices)
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-GB",
            name=voice_name,
        )
        response = self.client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": self.audio_config}
        )

        audio_bytes = io.BytesIO(response.audio_content).read()
        return base64.b64encode(audio_bytes).decode('utf-8')

# %%
