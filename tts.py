import os
from typing import IO
from io import BytesIO
import pygame
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

def text_to_speech_and_play(text: str):
    elevenlabs = ElevenLabs(
        api_key="sk_876917031f9deffa51bff1482a6938329ac67b5bb7dadc91",
    )
    
    # Perform the text-to-speech conversion
    response = elevenlabs.text_to_speech.stream(
        voice_id="pNInz6obpgDQGcFmaJgB",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )

    # Create a BytesIO object to hold the audio data in memory
    audio_stream = BytesIO()

    # Write each chunk of audio data to the stream
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)

    # Reset stream position to the beginning
    audio_stream.seek(0)

    # Play the audio
    pygame.mixer.init()
    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()
    
    # Wait for playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)

# Usage
text_to_speech_and_play("This is James")