#%%
from google.cloud import texttospeech
import io
import pygame
import random
client = texttospeech.TextToSpeechClient()

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=1
)
voices = ["en-GB-Journey-D",
"en-GB-News-K",
"en-GB-Wavenet-A",
"en-GB-Wavenet-F"]

pygame.mixer.init()
#%%
def text_to_speech(text):

    voice_name = voices[random.randint(0, 3)]
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-GB",
        name=voice_name,
    )
    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    audio_bytes = io.BytesIO(response.audio_content)
    pygame.mixer.music.load(audio_bytes)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
#%%