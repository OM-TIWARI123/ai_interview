from dotenv import load_dotenv
import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from graph2 import create_chat_graph, audio_streamer
from google import genai
import os
import time
import atexit

load_dotenv()

# Fix the API key issue
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

MONGODB_URI = "mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "2"}}

def cleanup():
    """Cleanup function to stop audio when exiting"""
    audio_streamer.stop()

# Register cleanup function
atexit.register(cleanup)

def main():
    print("Starting Silica Voice Agent...")
    print("Make sure your microphone is working and MongoDB is running.")
    
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)
        r = sr.Recognizer()
        
        # Adjust recognition settings for better performance
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        r.pause_threshold = 1.0
        r.operation_timeout = None
        r.phrase_threshold = 0.3
        r.non_speaking_duration = 0.8
        
        with sr.Microphone() as source:
            print("Calibrating microphone...")
            r.adjust_for_ambient_noise(source, duration=2)
            print("Calibration complete!")
            
            while True:
                try:
                    print("\n🎤 Listening... (speak now)")
                    audio = r.listen(source, timeout=1)
                    
                    print("🔄 Processing speech...")
                    try:
                        sst = r.recognize_google(audio)
                        print(f"👤 You said: {sst}")
                        
                        if sst.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                            print("👋 Goodbye!")
                            break
                        
                        print("🤖 Silica is responding...")
                        
                        # Stream the response
                        for event in graph.stream(
                            {"messages": [{"role": "user", "content": sst}]}, 
                            config, 
                            stream_mode="values"
                        ):
                            if "messages" in event:
                                # The streaming and audio generation happens in the chatbot node
                                # Just update the last message for context
                                pass
                        
                        # Small delay to ensure audio generation starts
                        time.sleep(0.5)
                        
                    except sr.UnknownValueError:
                        print("❌ Could not understand audio. Please speak clearly.")
                    except sr.RequestError as e:
                        print(f"❌ Could not request results from speech recognition service; {e}")
                    except Exception as e:
                        print(f"❌ Error processing speech: {e}")
                        
                except sr.WaitTimeoutError:
                    # Timeout is normal, just continue listening
                    pass
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
                    time.sleep(1)  # Brief pause before retrying

if __name__ == "__main__":
    main()