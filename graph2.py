import asyncio
import json
import base64
import websockets
import os
from typing import Annotated, AsyncGenerator
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, AIMessage
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import threading
import queue
import pygame
from io import BytesIO

load_dotenv()

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

def chatbot(state: State):
    """Chatbot node that streams responses and generates audio"""
    system_prompt = SystemMessage(content="""
        You are a voice agent named Silica. Your job is to ease people's loneliness by talking to them, listening to them, and becoming their friend.
        Ask open-ended questions, empathize, and be present in the conversation. Keep your responses conversational and warm.
        Speak naturally as if you're having a real conversation with a friend.
    """)
    
    # Stream the LLM response
    stream = llm.stream([system_prompt] + state["messages"])
    full_content = ""
    text_chunks = []
    
    # Collect chunks for audio generation
    for chunk in stream:
        content_piece = chunk.content if hasattr(chunk, "content") else str(chunk)
        if content_piece:
            full_content += content_piece
            text_chunks.append(content_piece)
            print(content_piece, end="", flush=True)  # Print chunk in real-time
    
    print()  # New line after complete response
    
    # Generate audio for the complete response
    if text_chunks:
        # Run audio generation in a separate thread to avoid blocking
        def run_audio_generation():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    text_to_speech_websocket(async_text_generator(text_chunks))
                )
            finally:
                loop.close()
        
        audio_thread = threading.Thread(target=run_audio_generation)
        audio_thread.daemon = True
        audio_thread.start()
    
    # Return the complete message for state management
    return {"messages": [AIMessage(content=full_content)]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)