import asyncio
import json
import base64
import websockets
import os
import threading
import queue
import pygame
import time
from io import BytesIO
from typing import Annotated, List, Dict, Any, AsyncGenerator
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from datetime import datetime
from threading import Event


class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# ElevenLabs configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
voice_id = 'Xb7hH8MSUJpSbSDYk0k2'  # You can change this to your preferred voice
model_id = 'eleven_flash_v2_5'

llm = init_chat_model("google_genai:gemini-2.0-flash")

class AudioStreamer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.audio_thread = None
        
    def start_audio_thread(self):
        """Start the audio playback thread"""
        if not self.is_playing:
            self.is_playing = True
            self.audio_thread = threading.Thread(target=self._audio_player)
            self.audio_thread.daemon = True
            self.audio_thread.start()
    
    def _audio_player(self):
        """Audio player thread that plays audio chunks as they arrive"""
        while self.is_playing:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                if audio_chunk is None:  # Sentinel to stop
                    break
                
                # Play the audio chunk
                audio_io = BytesIO(audio_chunk)
                pygame.mixer.music.load(audio_io)
                pygame.mixer.music.play()
                
                # Wait for the audio to finish playing
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")
    
    def add_audio_chunk(self, chunk):
        """Add an audio chunk to the playback queue"""
        self.audio_queue.put(chunk)
    
    def stop(self):
        """Stop the audio playback"""
        self.is_playing = False
        self.audio_queue.put(None)  # Sentinel to stop the thread

# Global audio streamer instance
audio_streamer = AudioStreamer()

async def text_to_speech_websocket(text_stream: AsyncGenerator[str, None]):
    """Stream text to ElevenLabs WebSocket and get audio chunks"""
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Initialize connection with voice settings
            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {
                    "stability": 0.5, 
                    "similarity_boost": 0.8, 
                    "use_speaker_boost": False
                },
                "generation_config": {
                    "chunk_length_schedule": [50, 120, 160, 290]
                },
                "xi_api_key": ELEVENLABS_API_KEY,
            }))
            
            # Start audio playback thread
            audio_streamer.start_audio_thread()
            
            # Create tasks for sending text and receiving audio
            send_task = asyncio.create_task(send_text_stream(websocket, text_stream))
            receive_task = asyncio.create_task(receive_audio_stream(websocket))
            
            # Wait for both tasks to complete
            await asyncio.gather(send_task, receive_task)
            
    except Exception as e:
        print(f"WebSocket error: {e}")

async def send_text_stream(websocket, text_stream: AsyncGenerator[str, None]):
    """Send streaming text to the WebSocket"""
    try:
        async for text_chunk in text_stream:
            if text_chunk:
                await websocket.send(json.dumps({"text": text_chunk}))
        
        # Send empty string to indicate end of text sequence
        await websocket.send(json.dumps({"text": "", "flush": True}))
        
    except Exception as e:
        print(f"Error sending text: {e}")

async def receive_audio_stream(websocket):
    """Receive audio chunks from the WebSocket"""
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data.get("audio"):
                # Decode audio and add to playback queue
                audio_chunk = base64.b64decode(data["audio"])
                audio_streamer.add_audio_chunk(audio_chunk)
                
            elif data.get('isFinal'):
                break
                
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"Error receiving audio: {e}")

async def async_text_generator(text_chunks):
    """Convert text chunks to async generator"""
    for chunk in text_chunks:
        yield chunk
        await asyncio.sleep(0.01)  # Small delay to allow for smooth streaming

class InterviewState(TypedDict):
    messages: Annotated[list, add_messages]
    role: str
    current_question_index: int
    questions_queue: List[Dict]
    responses: List[str]
    interview_phase: str  # "resume_processing", "question_generation", "asking_question", "waiting_response", "feedback"
    user_response: str
    is_generating:bool

