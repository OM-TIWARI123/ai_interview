from dotenv import load_dotenv
import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from graph import create_chat_graph
from google import genai
from google.genai import types
import wave # You'll need to install PyAudio
from typing import IO
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import os
load_dotenv()

client = genai.Client(api_key="GOOGLE_API_KEY")

MONGODB_URI = "mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "7"}}

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
elevenlabs = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)



def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)
        r=sr.Recognizer()

        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            r.pause_threshold=2
            while True:
                print("say something")
                audio=r.listen(source)

                print("processing audio...")
                sst=r.recognize_google(audio)

                print("you said: ",sst)

                for event in graph.stream({"messages":[{"role":"user","content":sst}]},config,stream_mode="values"):

                    if "messages" in event:
                        print("last message :",event["messages"][-1].content)
                        
                        event["messages"][-1].pretty_print()
                
main()